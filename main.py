from core.pool import AL_pool
from core.solver import solver
from core.data import total_dataset
from core.utils import print_n_txt
import torch
import random
import numpy as np

import argparse
import os
import logging

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str,default='mln',help='[base ,mln, mdn]')
parser.add_argument('--dataset', type=str,default='mnist',help='dataset_name')
parser.add_argument('--root', type=str,default='./dataset',help='root directory of the dataset')
parser.add_argument('--id', type=int,default=1,help='id')

parser.add_argument('--query_step', type=int,default=20,help='query step')
parser.add_argument('--query_size', type=int,default=20,help='query size')
parser.add_argument('--init_dataset', type=int,default=100,help='number of initial data')
parser.add_argument('--query_method', type=str,default='epistemic',help='query method')
parser.add_argument('--epoch', type=int,default=10,help='epoch')

parser.add_argument('--lr', type=float,default=1e-3,help='learning rate')
parser.add_argument('--batch_size', type=int,default=128,help='batch size')
parser.add_argument('--wd', type=float,default=1e-4,help='weight decay')

parser.add_argument('--lambda1', type=float,default=1,help='hyper parameter for mln loss')
parser.add_argument('--lambda2', type=float,default=1,help='hyper parameter for mln loss')

parser.add_argument('--k', type=int,default=10,help='number of mixtures')
parser.add_argument('--sig_min', type=float,default=1,help='sig min')
parser.add_argument('--sig_max', type=float,default=10,help='sig max')

args = parser.parse_args()

SEED = 1234
EPOCH = args.epoch

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device='cuda'

p = AL_pool(root=args.root,dataset_name=args.dataset,num_init=args.init_dataset)
test_dataset = total_dataset(root = args.root, dataset_name=args.dataset, train=False)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                        shuffle=False)

AL_solver = solver(args,device=device)
AL_solver.init_param()

train_iter,query_iter = p.subset_dataset(torch.zeros(size=(0,1),dtype=torch.int64).squeeze(1))

DIR = './res/{}_{}_{}/{}/'.format(args.mode,args.dataset,args.query_method,args.id)
try:
    os.mkdir('./res/{}_{}_{}'.format(args.mode,args.dataset,args.query_method))
except:
        pass
for i in range(args.query_step):
    try:
        os.mkdir(DIR)
    except:
        pass
    txtName = (DIR+'{}_log.txt'.format(i))
    f = open(txtName,'w') # Open txt file
    print_n_txt(_f=f,_chars='Text name: '+txtName)
    print_n_txt(_f=f,_chars=str(args))
    AL_solver.train_classification(train_iter,test_iter,f)
    id = AL_solver.query_data(query_iter)
    new = p.unlabled_idx[id]
    train_iter,query_iter = p.subset_dataset(new)