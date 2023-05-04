from transformers import BertTokenizer
from transformers.modeling_bert import BertForSequenceClassification,BertForMaskedLM
#from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead,BertForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import glue_convert_examples_to_features as convert_examples_to_features
import torch.nn as nn
import torch

from tqdm import tqdm
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

BERT_MODEL = 'bert-base-uncased'


class AWAClassifier:
    def __init__(self, label_list, device, cache_dir, gamma, adv_rd, fix_adv=False):
        self._label_list = label_list
        self._device = device
        self._fix_adv = fix_adv
        self._rd = adv_rd if not fix_adv else 0
        self.gamma = gamma

        self._tokenizer = BertTokenizer.from_pretrained(BERT_MODEL,
                                                        do_lower_case=True,
                                                        cache_dir=cache_dir)

        self._model = BertForSequenceClassification.from_pretrained(BERT_MODEL,
                                                                    num_labels=len(label_list),
                                                                    cache_dir=cache_dir,
                                                                    output_hidden_states=True)
        self._model.to(device)

        self._optimizer = None


        self.adv_net = nn.ModuleList([nn.Linear(768, 1) for _ in range(len(label_list))])
        self.adv_net.to(device)
        
            
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

    def get_optimizer(self, learning_rate, adv_lr):
        self._optimizer = _get_optimizer(
            self._model, learning_rate=learning_rate)
        self._adv_opt = _get_optimizer(
            self.adv_net, learning_rate=adv_lr)

    def save(self, save_dir):
        torch.save({'model': self._model, 'adv': self.adv_net}, save_dir)
    
    def unfreeze_adv(self):
        self.adv_net.requires_grad = True

    def train_epoch(self):
        self._model.train()

        for step, batch in enumerate(tqdm(self._data_loader['train'],
                                          desc='Training')):
            batch = tuple(t.to(self._device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            sentence_length = (batch[1]).sum(dim=-1).detach()
            # print(sentence_length)
            outputs = self._model(**inputs)
            self._optimizer.zero_grad()
            loss = outputs[0]
            loss.backward()
            self._optimizer.step()
            # adv

            # update adv
            unk_id = self._tokenizer.encode(('UNK'), return_tensors='pt').to(self._device)
            for _ in range(self._rd):

                # get rid of gradient
                encoder = self._model.get_input_embeddings().to(self._device)
                unk_embeddings = encoder(unk_id)[:, 1, :]
                bert_embedding = encoder(batch[0])
                
                alpha_list = []
                for ix in range(batch[0].size(0)):
                    alpha_list.append(self.adv_net[batch[3][ix]](bert_embedding[ix]).unsqueeze(0))
                    # alpha_list.append(self.adv_net[0](bert_embedding[ix]).unsqueeze(0))
                alpha = torch.sigmoid(torch.cat(alpha_list, dim=0)) # label specific
                # print(alpha.squeeze()[0,:5])
                reg_loss = torch.mean(torch.sum(alpha.squeeze(-1), dim=1))
                bert_embedding = (1 - alpha) * bert_embedding + alpha * unk_embeddings
                input_adv = {
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'inputs_embeds': bert_embedding,
                    'labels': batch[3]
                }
                outputs_adv = self._model(**input_adv)[0]
                self._adv_opt.zero_grad()
                loss = -outputs_adv + self.gamma * reg_loss # maximize
                loss.backward()
                self._adv_opt.step()

            # generate adv train
            encoder = self._model.get_input_embeddings().to(self._device)
            # get rid of gradient
            unk_embeddings = encoder(unk_id)[:, 1, :].detach()
            bert_embedding = encoder(batch[0])
            
            alpha_list = []
            for ix in range(batch[0].size(0)):
                alpha_list.append(self.adv_net[batch[3][ix]](bert_embedding[ix]).unsqueeze(0))
            alpha = torch.sigmoid(torch.cat(alpha_list, dim=0)).detach() # label specific
            bert_embedding = (1 - alpha) * bert_embedding + alpha * unk_embeddings
            input_adv = {
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'inputs_embeds': bert_embedding,
                'labels': batch[3]
            }
            outputs_adv = self._model(**input_adv)[0]
            
            self._optimizer.zero_grad()
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
