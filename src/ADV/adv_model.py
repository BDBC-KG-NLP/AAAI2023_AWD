from transformers import BertTokenizer
from transformers.modeling_bert import BertForSequenceClassification,BertForMaskedLM
#from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead,BertForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import glue_convert_examples_to_features as convert_examples_to_features
import torch.nn as nn
import torch

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset

BERT_MODEL = 'bert-base-uncased'


class Classifier:
    def __init__(self, label_list, device, cache_dir, attacker=None, epsilon=None):
        self._label_list = label_list
        self._device = device
        self._epsilon = epsilon
        self.attacker = attacker

        self._tokenizer = BertTokenizer.from_pretrained(BERT_MODEL,
                                                        do_lower_case=True,
                                                        cache_dir=cache_dir)

        self._model = BertForSequenceClassification.from_pretrained(BERT_MODEL,
                                                                    num_labels=len(label_list),
                                                                    cache_dir=cache_dir)
        self._model.to(device)

        self._optimizer = None


        self._dataset = {}
        self._data_loader = {}

    def load_data(self, set_type, examples, batch_size, max_length, shuffle):
        self._dataset[set_type] = examples
        self._data_loader[set_type] = _make_data_loader(
            examples=examples,
            label_list=self._label_list,
            tokenizer=self._tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=shuffle)

    def get_optimizer(self, learning_rate):
        self._optimizer = _get_optimizer(
            self._model, learning_rate=learning_rate)
    
    def save(self, save_dir):
        torch.save({'model': self._model}, save_dir)


    def train_epoch(self):
        self._model.train()

        for step, batch in enumerate(tqdm(self._data_loader['train'],
                                          desc='Training')):
            batch = tuple(t.to(self._device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}

            self._optimizer.zero_grad()
            outputs = self._model(**inputs)
            loss = outputs[0]  
            loss.backward()
            self._optimizer.step()

            # adv

            # update adv
            

            encoder = self._model.get_input_embeddings().to(self._device)
            bert_embedding = encoder(batch[0])
            
            
            input_adv = {
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'inputs_embeds': bert_embedding,
                'labels': batch[3]
            }
            adv_embedding, ce_loss = self.attacker(input_adv, batch[3], self._model, self._epsilon)
            input_adv = {
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'inputs_embeds': adv_embedding,
                'labels': batch[3]
            }
            self._optimizer.zero_grad()
            outputs_adv = self._model(**input_adv)[0]
            print('ce_loss:', ce_loss, 'adv_loss:', outputs_adv)
            loss = outputs_adv
            loss.backward()
            self._optimizer.step()


    def evaluate(self, set_type):
        self._model.eval()

        preds_all, labels_all = [], []
        data_loader = self._data_loader[set_type]

        for batch in tqdm(data_loader,
                          desc="Evaluating {} set".format(set_type)):
            batch = tuple(t.to(self._device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}

            with torch.no_grad():
                outputs = self._model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
            preds = torch.argmax(logits, dim=1)

            preds_all.append(preds)
            labels_all.append(inputs["labels"])

        preds_all = torch.cat(preds_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)

        return torch.sum(preds_all == labels_all).item() / labels_all.shape[0]


def _get_optimizer(model, learning_rate):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    return optimizer


def _make_data_loader(examples, label_list, tokenizer, batch_size, max_length, shuffle):
    features = convert_examples_to_features(examples,
                                            tokenizer,
                                            label_list=label_list,
                                            max_length=max_length,
                                            output_mode="classification")
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
