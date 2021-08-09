import torch
from core.MLN.loss import *
import matplotlib.pyplot as plt
import numpy as np

def test_eval(model,data_iter,data_size,device):
    with torch.no_grad():
        n_total,n_correct,epis_unct_sum,alea_unct_sum,entropy_pi_sum = 0,0,0,0,0
        y_probs= list()
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in,batch_out in data_iter:
            # Foraward path
            y_trgt      = batch_out.to(device)
            if data_size is None:
                mln_out     = model.forward(batch_in.to(device))
            else:
                mln_out     = model.forward(batch_in.view(data_size).to(device))
            pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
            out         = mln_gather(pi,mu,sigma)
            model_pred  = out['mu_sel'] # [B x N]

            unct_out    = mln_uncertainties(pi,mu,sigma)
            epis_unct   = unct_out['epis'] # [N]
            alea_unct   = unct_out['alea'] # [N]            entropy_pi  = -pi*torch.log(pi)
            entropy_pi  = unct_out['pi_entropy']
            entropy_pi_sum  += torch.sum(entropy_pi)
            epis_unct_sum += torch.sum(epis_unct)
            alea_unct_sum += torch.sum(alea_unct)
            # Check predictions
            y_prob,y_pred    = torch.max(model_pred,1)
            n_correct   += (y_pred==y_trgt).sum().item()
            #print(y_trgt)
            n_total     += batch_in.size(0)
            
            y_probs += list(y_prob.cpu().numpy())
            
        val_accr  = (n_correct/n_total)
        entropy_pi_avg=(entropy_pi_sum/n_total).detach().cpu().item()
        epis      = (epis_unct_sum/n_total).detach().cpu().item()
        alea      = (alea_unct_sum/n_total).detach().cpu().item()
        model.train() # back to train mode 
        out_eval = {'val_accr':val_accr,'epis':epis,'alea':alea,
                    'pi_entropy':entropy_pi_avg}
        model.train() # back to train mode 
    return out_eval

def func_eval(model,data_iter,data_size,device):
    with torch.no_grad():
        epis_unct_sum,alea_unct_sum,n_total = 0,0,0
        epis_ = list()
        alea_ = list()
        pi_entropy_ = list()
        maxsoftmax_ = list()
        entropy_ = list()
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in in data_iter:
            # Foraward path
            mln_out     = model.forward(batch_in.view(data_size).to(device))
            pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
            out         = mln_gather(pi,mu,sigma)
            model_pred  = out['mu_sel'] # [B x N]

            #print(pi)
            # Compute uncertainty 
            unct_out    = mln_uncertainties(pi,mu,sigma)
            epis_unct   = unct_out['epis'] # [N]
            alea_unct   = unct_out['alea'] # [N]
            pi_entropy  = unct_out['pi_entropy']
            
            epis_unct_sum += torch.sum(epis_unct)
            alea_unct_sum += torch.sum(alea_unct)
            
            y_prob,_ = torch.max(model_pred,1)
            
            entropy = -y_prob*torch.log(y_prob)

            maxsoftmax_ += list(1-y_prob.cpu().numpy())
            epis_ += list(epis_unct.cpu().numpy())
            alea_ += list(alea_unct.cpu().numpy())
            pi_entropy_ += list(pi_entropy.cpu().numpy())
            entropy_    += list(entropy.cpu().numpy())
            n_total     += batch_in.size(0)

        epis      = (epis_unct_sum/n_total).detach().cpu().item()
        alea      = (alea_unct_sum/n_total).detach().cpu().item()
        model.train() # back to train mode 
        out_eval = {'epis':epis,'alea':alea, 
                        'epis_' : epis_,'alea_' : alea_, 'maxsoftmax_':maxsoftmax_,
                        'pi_entropy_':pi_entropy_,'entropy_':entropy_}
    return out_eval