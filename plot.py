import matplotlib.pyplot as plt
import os,json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str,default='mln',help='[base ,mln, mdn]')
parser.add_argument('--dataset', type=str,default='mnist',help='dataset_name')

parser.add_argument('--id', type=int,default=1,help='id')

args = parser.parse_args()

if args.mode == 'mln':
    method = ['epistemic','aleatoric','pi_entropy','maxsoftmax','entropy','random']
elif args.mode == 'base':
    method = ['maxsoftmax','entropy','coreset','random']
elif args.mode == 'bald':
    method = ['maxsoftmax','entropy','bald','mean_std','random']
else:
    raise NotImplementedError

test_acc={}
train_acc = {}
for m in method:
    DIR = './res/{}_{}_{}/{}/log.json'.format(args.mode,args.dataset,m,args.id)
    with open(DIR) as f:
        data = json.load(f)
    test_acc[m] = data['test_accr']
    train_acc[m] = data['train_accr']
    f.close()

plt.figure()
plt.title("{} {} Active Learning Test Accuracy".format(args.mode,args.dataset))
for m in method:
    plt.plot(test_acc[m],label=m)
plt.xlabel("Query Step")
plt.legend()
plt.savefig("./res/{}_{}_{}.png".format(args.mode,args.dataset,args.id))
#plt.show()