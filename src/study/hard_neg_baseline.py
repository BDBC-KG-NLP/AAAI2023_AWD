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

    classifier = AWAClassifier(label_list=label_list, device=device, cache_dir=args.cache, reg=0, adv_rd=0, fix_adv=True)
    classifier.get_optimizer(learning_rate=args.learning_rate, adv_lr=0.1)

    classifier.load_data(
        'train', examples['train'], args.batch_size, max_length=args.max_seq_length, shuffle=True)
    classifier.load_data(
        'dev', examples['dev'], args.batch_size, max_length=args.max_seq_length, shuffle=False)
    classifier.load_data(
        'test', examples['test'], args.batch_size, max_length=args.max_seq_length, shuffle=False)

    rec = 1
    for i in range(30):
        if os.path.exists(os.path.join(args.model_dir, f'model_{i+1}.pth')):
            rec = i+1
    model_dir = os.path.join(args.model_dir, f'model_{rec}.pth')
    group_dir = os.path.join(args.model_dir, '..', '..')
    state_dict = torch.load(model_dir)
    classifier._model = state_dict['model']

    acc = classifier.evaluate('train')
    with open(os.path.join(group_dir, f'train_hard_neg_{args.name}.log'), 'a+') as f:
        f.write(f'{(100. * acc):.4f}\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['stsa', 'snips', 'trec'])
    parser.add_argument('--data_dir', type=str, help="Data dir path with {train, dev, test}.tsv")
    parser.add_argument('--model_dir', type=str, help="Data dir path with {train, dev, test}.tsv")
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

    args = parser.parse_args()
    print(args)
    main(args)

