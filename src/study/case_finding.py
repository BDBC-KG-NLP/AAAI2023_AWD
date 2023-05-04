import torch, os
import argparse

from data_processors import get_data
from AWA_model import AWAClassifier
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    examples, label_list = get_data(
        task=args.task,
        data_dir=args.data_dir,
        data_seed=args.seed)

    classifier = AWAClassifier(label_list=label_list, device=device, cache_dir=args.cache, reg=args.reg_l, adv_rd=args.adv_iter, gamma=args.gamma, fix_adv=True)
    classifier.get_optimizer(learning_rate=args.learning_rate, adv_lr=args.adv_lr)

    classifier.load_data(
        'train', examples['train'], args.batch_size, max_length=args.max_seq_length, shuffle=False)
    classifier.load_data(
        'dev', examples['dev'], args.batch_size, max_length=args.max_seq_length, shuffle=False)
    classifier.load_data(
        'test', examples['test'], args.batch_size, max_length=args.max_seq_length, shuffle=False)


    save_dir = os.path.join(args.data_dir, args.name)
    rec = 1
    for i in range(30):
        if os.path.exists(os.path.join(save_dir, f'model_{i+1}.pth')):
            rec = i+1
    model_dir = os.path.join(save_dir, f'model_{rec}.pth')
    state_dict = torch.load(model_dir)
    classifier.adv_net = state_dict['adv']

    for step, batch in enumerate(classifier._data_loader['train']):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels': batch[3]}
        for ix in range(batch[0].size(0)):
            input_ids = batch[0][ix];
            print(input_ids)
            tokens = []
            for idx in range(input_ids.size(0)):
                dec = classifier._tokenizer.decode([input_ids[idx]])
                tokens.append(dec)
            print(tokens)
            unk_id = classifier._tokenizer.encode(('UNK'), return_tensors='pt').to(device)
            # generate adv train
            encoder = classifier._model.get_input_embeddings().to(device)
            # get rid of gradient
            unk_embeddings = encoder(unk_id)[:, 1, :]
            bert_embedding = encoder(input_ids)
            
            alpha_list = []
            alpha_list.append(classifier.adv_net[batch[3][ix]](bert_embedding).unsqueeze(0))
            alpha = torch.sigmoid(torch.cat(alpha_list, dim=0)) # label specific
            print(alpha.squeeze(-1))
            print('label:', batch[3][ix])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['stsa', 'snips', 'trec'])
    parser.add_argument('--data_dir', type=str, help="Data dir path with {train, dev, test}.tsv")
    parser.add_argument('--seed', default=159, type=int)
    parser.add_argument('--hidden_dropout_prob', default=0.1, type=float)
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, sequences shorter will be padded.")

    parser.add_argument('--cache', default="transformers_cache", type=str)

    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument("--learning_rate", default=4e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--name', type=str)
    parser.add_argument('--adv_lr', default=0.1, type=float)
    parser.add_argument('--reg_l', default=0.01, type=float)
    parser.add_argument('--adv_iter', default=1, type=int)
    parser.add_argument('--gamma', default=1, type=float)

    args = parser.parse_args()
    print(args)
    print(233)
    main(args)

