import torch
import torch.optim as optim
from core.MLN.model import MixtureLogitNetwork_cnn
from core.MLN.loss import mace_loss
from core.MLN.eval import func_eval,test_eval
from core.utils import print_n_txt

class solver():
    def __init__(self,args,device):
        self.EPOCH = args.epoch
        self.mode_name = args.mode
        if args.dataset == 'mnist':
            self.data_size = (-1,1,28,28)
            self.labels=10
        elif args.dataset == 'cifar10':
            self.data_size = (-1,3,32,32)
            self.labels=10
        self.device = device
        self.load_model(args)
        self.optimizer = optim.Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4,eps=1e-8)
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.method = args.query_method
        self.query_size = args.query_size

    def load_model(self,args):
        if self.mode_name == 'mln':
            self.model = MixtureLogitNetwork_cnn(name='mln',x_dim=[1,28,28],k_size=3,c_dims=[32,64,128],p_sizes=[2,2,2],
                            h_dims=[128,64],y_dim=self.labels,USE_BN=False,k=args.k,
                            sig_min=args.sig_min,sig_max=args.sig_max, 
                            mu_min=-1,mu_max=+1,SHARE_SIG=True).to(self.device)
    def init_param(self):
        self.model.init_param()

    def train_classification(self,train_iter,test_iter,f):
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120,150,180], gamma=args.lr_rate)
        #scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=0.2, step_size=20)
        for epoch in range(self.EPOCH):
            loss_sum = 0.0
            #time.sleep(1)
            for batch_in,batch_out in train_iter:
                mln_out = self.model.forward(batch_in.view(self.data_size).to(self.device))
                pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
                target = torch.eye(self.labels)[batch_out].to(self.device)
                target=target.to(self.device)
                loss_out = mace_loss(pi,mu,sigma,target) # 'mace_avg','epis_avg','alea_avg'
                loss = loss_out['mace_avg'] - self.lambda1 * loss_out['epis_avg'] + self.lambda2 * loss_out['alea_avg']
                #print(loss)
                self.optimizer.zero_grad() # reset gradient
                loss.backward() # back-propagation
                self.optimizer.step() # optimizer update
                # Track losses
                loss_sum += loss
            #scheduler.step()
            loss_avg = loss_sum/len(train_iter)
            test_out = test_eval(self.model,test_iter,self.data_size,'cuda')
            train_out = test_eval(self.model,train_iter,self.data_size,'cuda')

            strTemp = ("epoch: [%d/%d] loss: [%.3f] train_accr:[%.4f] test_accr: [%.4f]"
                        %(epoch,self.EPOCH,loss_avg,train_out['val_accr'],test_out['val_accr']))
            print_n_txt(_f=f,_chars=strTemp)

            strTemp =  ("[Train] mace_avg:[%.4f] epis avg:[%.3f] alea avg:[%.3f] pi_entropy avg: [%.3f]"%
                (loss_out['mace_avg'],loss_out['epis_avg'],loss_out['alea_avg'],loss_out['pi_entropy_avg']))
            print_n_txt(_f=f,_chars=strTemp)

            strTemp =  ("[Test] epis avg:[%.3f] alea avg:[%.3f] pi_entropy avg: [%.3f]"%
                    (test_out['epis'],test_out['alea'],test_out['pi_entropy']))
            print_n_txt(_f=f,_chars=strTemp)
        return train_out['val_accr'],test_out['val_accr']
    
    def query_data(self,infer_iter):
        out = func_eval(self.model,infer_iter,self.data_size,'cuda')
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
        else:
            raise NotImplementedError()
        out  = torch.FloatTensor(out)
        _, max_idx = torch.topk(out,self.query_size,0)
        max_idx = max_idx.type(torch.LongTensor)
        return max_idx