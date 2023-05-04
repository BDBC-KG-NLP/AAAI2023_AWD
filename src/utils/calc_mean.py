import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
args = parser.parse_args()
with open(args.dir) as f:
    res = []
    for line in f:
        for x in line.split():
            res.append(float(x))
with open(args.dir, 'a+') as f:
    print(sum(res)/len(res), file=f)

