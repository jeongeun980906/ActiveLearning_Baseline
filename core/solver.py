import torch
import torch.optim as optim
import torch.nn as nn
import random

from core.MLN.model import MixtureLogitNetwork_cnn,MixtureLogitNetwork_cnn2
from core.MLN.loss import mace_loss
from core.MLN.eval import func_eval_mln,test_eval_mln
from core.utils import print_n_txt,print_log_baseline,print_log_bald

from core.baseline.model import CNN7,CNN3
from core.baseline.eval import func_eval_baseline,test_eval_baseline
from core.baseline.bald_eval import func_eval_bald, test_eval_bald
from core.baseline.coreset import func_eval_coreset,coreset

class solver():
    def __init__(self,args,device):
        self.EPOCH = args.epoch
        self.mode_name = args.mode
        self.dataset = args.dataset
        if args.dataset == 'mnist':
            self.data_config = [(-1,1,28,28),10]
            self.labels=10
        elif args.dataset == 'cifar10':
            self.data_config = [(-1,3,32,32),10]
            self.labels=10
        self.device = device
        self.load_model(args)
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.method = args.query_method
        self.query_size = args.query_size
        self.query_init_weight = args.init_weight
        self.get_function()

    def load_model(self,args):
        if self.dataset == 'mnist':
            if self.mode_name == 'mln':
                self.model = MixtureLogitNetwork_cnn(name='mln',x_dim=[1,28,28],k_size=3,c_dims=[32,64,128],p_sizes=[2,2,2],
                            h_dims=[128,64],y_dim=self.labels,USE_BN=False,k=args.k,
                            sig_min=args.sig_min,sig_max=args.sig_max, 
                            mu_min=-1,mu_max=+1,SHARE_SIG=True).to(self.device)
    
            elif self.mode_name == 'base' or self.mode_name == 'bald':
                self.model = CNN3().to(self.device)
            else:
                raise NotImplementedError
        elif self.dataset == 'cifar10':
            if self.mode_name == 'mln':
                self.model = MixtureLogitNetwork_cnn2(name='mln',x_dim=[3,32,32],c_dims = [64,64,128,128,196,16],h_dims=[],
                            p_sizes= [2,2,2], k_size=3,y_dim=self.labels,USE_BN=True,k=args.k,
                            sig_min=args.sig_min,sig_max=args.sig_max, 
                            mu_min=-3,mu_max=+3,SHARE_SIG=True).to(self.device)
    
            elif self.mode_name == 'base' or self.mode_name == 'bald':
                self.model = CNN7().to(self.device)
            else:
                raise NotImplementedError            

    def get_function(self):
        if self.mode_name == 'mln':
            self.train = self.train_mln
            self.test = test_eval_mln
            self.query_function = func_eval_mln
        elif self.mode_name == 'base':
            self.train = self.train_base
            self.test = test_eval_baseline
            if self.method == 'coreset':
                self.query_function = func_eval_coreset
            else:
                self.query_function = func_eval_baseline
            self.print_function = print_log_baseline
        elif self.mode_name == 'bald':
            self.train = self.train_base
            self.test = test_eval_bald
            self.query_function = func_eval_bald
            self.print_function = print_log_bald

    def init_param(self):
        self.model.init_param()

    def train_mln(self,train_iter,test_iter,ltrain,ltest,f):
        if self.query_init_weight:
            self.init_param()
        optimizer = optim.Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4,eps=1e-8)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120,150,180], gamma=args.lr_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size=50)
        for epoch in range(self.EPOCH):
            loss_sum = 0.0
            #time.sleep(1)
            for batch_in,batch_out in train_iter:
                mln_out = self.model.forward(batch_in.view(self.data_config[0]).to(self.device))
                pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
                target = torch.eye(self.labels)[batch_out].to(self.device)
                target=target.to(self.device)
                loss_out = mace_loss(pi,mu,sigma,target) # 'mace_avg','epis_avg','alea_avg'
                loss = loss_out['mace_avg'] - self.lambda1 * loss_out['epis_avg'] + self.lambda2 * loss_out['alea_avg']
                #print(loss)
                optimizer.zero_grad() # reset gradient
                loss.backward() # back-propagation
                optimizer.step() # optimizer update
                # Track losses
                loss_sum += loss
            scheduler.step()
            loss_avg = loss_sum/len(train_iter)
            test_out = self.test(self.model,test_iter,self.data_config,ltest,'cuda')
            train_out = self.test(self.model,train_iter,self.data_config,ltrain,'cuda')

            strTemp = ("epoch: [%d/%d] loss: [%.3f] train_accr:[%.4f] test_accr: [%.4f]"
                        %(epoch,self.EPOCH,loss_avg,train_out['val_accr'],test_out['val_accr']))
            print_n_txt(_f=f,_chars=strTemp)

            strTemp =  ("[Train] mace_avg: [%.4f] epis avg: [%.3f] alea avg: [%.3f] pi_entropy avg: [%.3f]"%
                (loss_out['mace_avg'],loss_out['epis_avg'],loss_out['alea_avg'],loss_out['pi_entropy_avg']))
            print_n_txt(_f=f,_chars=strTemp)

            strTemp =  ("[Test] epis avg: [%.3f] alea avg: [%.3f] pi_entropy avg: [%.3f]"%
                    (test_out['epis'],test_out['alea'],test_out['pi_entropy']))
            print_n_txt(_f=f,_chars=strTemp)
        return train_out['val_accr'],test_out['val_accr']
    
    def train_base(self,train_iter,test_iter,ltrain,ltest,f):
        criterion = nn.CrossEntropyLoss().cuda()
        if self.query_init_weight:
            self.init_param()
        optimizer = optim.Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4,eps=1e-8)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120,150,180], gamma=args.lr_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size=50)
        for epoch in range(self.EPOCH):
            loss_sum = 0.0
            #time.sleep(1)
            for batch_in,batch_out in train_iter:
                output = self.model.forward(batch_in.view(self.data_config[0]).to(self.device))
                target = batch_out.to(self.device)
                loss = criterion(output,target)
                #print(loss)
                optimizer.zero_grad() # reset gradient
                loss.backward() # back-propagation
                optimizer.step() # optimizer update
                # Track losses
                loss_sum += loss.item()
            scheduler.step()
            loss_avg = loss_sum/len(train_iter)
            test_out = self.test(self.model,test_iter,self.data_config,ltest,'cuda')
            train_out = self.test(self.model,train_iter,self.data_config,ltrain,'cuda')
            self.print_function(f,epoch,self.EPOCH,loss_avg,train_out,test_out)
        return train_out['val_accr'],test_out['val_accr']

    def query_data(self,unlabel_iter,label_iter,unl_size=None,l_size=None):
        out = self.query_function(self.model,unlabel_iter,label_iter,self.data_config,unl_size,l_size,'cuda')
        if self.method == 'epistemic':
            out = out['epis_']
        elif self.method == 'aleatoric':
            out = out['alea_']
        elif self.method == 'maxsoftmax':
            out = out['maxsoftmax_']
        elif self.method == 'entropy':
            out = out['entropy_']
        elif self.method == 'pi_entropy':
            out = out['pi_entropy_']
        elif self.method == 'coreset':
            c = coreset(out['labeled'],out['unlabeled'])
            return c.robust_k_center(self.query_size)
        elif self.method == 'mean_std':
            out = out['mean_std_']
        elif self.method == 'bald':
            out = out['bald_']
        elif self.method == 'random':
            return  torch.tensor(random.sample(range(unl_size), self.query_size))
        else:
            raise NotImplementedError()
        out  = torch.FloatTensor(out)
        _, max_idx = torch.topk(out,self.query_size,0)
        max_idx = max_idx.type(torch.LongTensor)
        return max_idx