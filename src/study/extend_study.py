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
        'train', examples['train'], args.batch_size, max_length=args.max_seq_length, shuffle=True)
    classifier.load_data(
        'dev', examples['dev'], args.batch_size, max_length=args.max_seq_length, shuffle=False)
    classifier.load_data(
        'test', examples['test'], args.batch_size, max_length=args.max_seq_length, shuffle=False)

    save_dir = os.path.join(args.model_dir, args.name)
    rec = 1
    for i in range(30):
        if os.path.exists(os.path.join(save_dir, f'model_{i+1}.pth')):
            rec = i+1
    model_dir = os.path.join(save_dir, f'model_{rec}.pth')
    state_dict = torch.load(model_dir)
    classifier.adv_net = state_dict['adv']
    group_dir = os.path.join(args.data_dir, '..')
    best_dev_acc, final_test_acc = -1., -1.
    for epoch in range(args.epochs):
        classifier.train_epoch()
        dev_acc = classifier.evaluate('dev')

        do_test = (dev_acc > best_dev_acc)
        best_dev_acc = max(best_dev_acc, dev_acc)

        print('Epoch {}, Dev Acc: {:.4f}, Best Ever: {:.4f}'.format(
            epoch, 100. * dev_acc, 100. * best_dev_acc))

        if do_test:
            final_test_acc = classifier.evaluate('test')
            print('Test Acc: {:.4f}'.format(100. * final_test_acc))
        with open(os.path.join(save_dir, 'transfer_result.txt'), 'a+') as f:
            f.write(f'{(100. * final_test_acc):.4f}\t')


    print('Final Dev Acc: {:.4f}, Final Test Acc: {:.4f}'.format(
        100. * best_dev_acc, 100. * final_test_acc))
    with open(os.path.join(group_dir, f'transfer_{args.name}.log'), 'a+') as f:
        f.write(f'{(100. * final_test_acc):.4f}\t')



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
    parser.add_argument('--adv_lr', default=0.1, type=float)
    parser.add_argument('--reg_l', default=0.01, type=float)
    parser.add_argument('--adv_iter', default=1, type=int)
    parser.add_argument('--gamma', default=1, type=float)

    args = parser.parse_args()
    print(args)
    main(args)

