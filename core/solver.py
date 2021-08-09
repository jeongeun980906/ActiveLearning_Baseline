import torch
import torch.optim as optim
from core.MLN.model import MixtureLogitNetwork_cnn
from core.MLN.loss import mace_loss
from core.MLN.eval import func_eval,test_eval

class solver():
    def __init__(self):
        self.EPOCH = 10
        self.mode_name = 'mln'
        self.data_size = (-1,1,28,28)
        self.device = 'cuda'
        self.init_model()
        self.optimizer = optim.Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4,eps=1e-8)
        self.labels=10
    def init_model(self):
        if self.mode_name == 'mln':
            self.model = MixtureLogitNetwork_cnn(name='mln',x_dim=[1,28,28],k_size=3,c_dims=[32,64,128],p_sizes=[2,2,2],
                            h_dims=[128,64],y_dim=10,USE_BN=False,k=10,
                            sig_min=1.0,sig_max=10, 
                            mu_min=-1,mu_max=+1,SHARE_SIG=True).to(self.device)
    def init_param(self):
        self.model.init_param()

    def train_image(self,train_iter):
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120,150,180], gamma=args.lr_rate)
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=0.2, step_size=5)
        for epoch in range(self.EPOCH):
            loss_sum = 0.0
            #time.sleep(1)
            for batch_in,batch_out in train_iter:
                mln_out = self.model.forward(batch_in.view(self.data_size).to(self.device))
                pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
                target = torch.eye(self.labels)[batch_out].to(self.device)
                target=target.to(self.device)
                loss_out = mace_loss(pi,mu,sigma,target) # 'mace_avg','epis_avg','alea_avg'
                loss = loss_out['mace_avg'] - loss_out['epis_avg'] + loss_out['alea_avg']
                #print(loss)
                self.optimizer.zero_grad() # reset gradient
                loss.backward() # back-propagation
                self.optimizer.step() # optimizer update
                # Track losses
                loss_sum += loss
            #scheduler.step()
            loss_avg = loss_sum/len(train_iter)
            print(loss_avg.item())
    
    def query_data(self,infer_iter, method='epistemic',query_number=20):
        out = func_eval(self.model,infer_iter,self.data_size,'cuda')
        if method == 'epistemic':
            out = out['epis_']
        elif method == 'aleatoric':
            out = out['alea_']
        out  = torch.FloatTensor(out)
        _, max_idx = torch.topk(out,query_number,0)
        max_idx = max_idx.type(torch.LongTensor)
        return max_idx
    
    def test_acc(self,test_iter):
        out = test_eval(self.model,test_iter,self.data_size,'cuda')
        print("ACC: {}".format(out['val_accr']))