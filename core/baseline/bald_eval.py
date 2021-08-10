import torch
import numpy as np
'''
BALD, Mean STD
'''
def test_eval_bald(model,data_iter,data_size,device):
    total=10
    prob,prob_ = list(),list()
    for T in range(total):
        prob = []
        target=[]
        with torch.no_grad():
            n_total = 0        
            for batch_in,batch_out in data_iter:
                # Foraward path
                model_pred    = model.forward(batch_in.view(data_size).to(device))
                model_pred = torch.softmax(model_pred,1)
                prob +=list(model_pred.cpu().numpy())
                n_total     += batch_in.size(0)
                target += list(batch_out.cpu().numpy())
            prob_ += [prob]
    prob = torch.FloatTensor(prob_) # [T x N x D]
    target= torch.LongTensor(target) # [N]
    mean = torch.mean(prob,dim=0) # [N x D]
    # maxsoftmax
    maxsoftmax,y_pred = torch.max(mean,dim=1) # [N]
    variation_ratio = 1-maxsoftmax # [N]
    maxsoftmax_avg = torch.mean(maxsoftmax).cpu().item() # [1]
    # Correct
    val_accr = ((target == y_pred).sum()/n_total).cpu().item() # [1]
    # BALD, entropy
    total_entropy = torch.sum(-mean*torch.log(mean+1e-6),dim=-1) # [N]
    total_entropy_avg = torch.mean(total_entropy,dim=0).cpu().item() # [1]
    entropy = torch.sum(-prob*torch.log(prob+1e-6),dim=-1) # [T x N]
    avg_entropy = torch.mean(entropy,dim=0) # [N]
    mutual_information = total_entropy - avg_entropy # [N]
    mutual_information_avg = torch.mean(mutual_information,dim=0).item() # [1]

    # Mean STD
    var = torch.mean(torch.square(prob+1e-6),dim=0)-torch.square(mean+1e-6) # [N x D]
    mean_std = torch.mean(torch.sqrt(var+1e-6),dim=-1) # [N]
    # print((torch.isnan(var)==True).sum())
    mean_std_avg = torch.mean(mean_std,dim=0).cpu().item() # [1]
    out_eval = {'val_accr':val_accr,'maxsoftmax':maxsoftmax_avg,'entropy':total_entropy_avg,
                'bald':mutual_information_avg,'mean_std':mean_std_avg}
    return out_eval

def func_eval_bald(model,data_iter,data_size,device):
    total=10
    prob,prob_ = list(),list()
    for T in range(total):
        prob = []
        with torch.no_grad():
            n_total = 0        
            for batch_in in data_iter:
                # Foraward path
                model_pred    = model.forward(batch_in.view(data_size).to(device))
                model_pred = torch.softmax(model_pred,1)
                prob +=list(model_pred.cpu().numpy())
                n_total     += batch_in.size(0)
            prob_ += [prob]
    prob = torch.FloatTensor(prob_) # [T x N x D]
    mean = torch.mean(prob,dim=0) # [N x D]
    # maxsoftmax
    maxsoftmax,y_pred = torch.max(mean,dim=1)
    variation_ratio = 1-maxsoftmax # [N]

    # BALD, Entropy
    total_entropy = torch.sum(-mean*torch.log(mean+1e-6),dim=-1) # [N]
    entropy = torch.sum(-prob*torch.log(prob+1e-6),dim=-1) # [T x N]
    avg_entropy = torch.mean(entropy,dim=0) # [N]
    mutual_information = total_entropy - avg_entropy
    var = torch.mean(torch.square(prob+1e-6),dim=0)-torch.square(mean+1e-6) # [N x D]
    mean_std = torch.mean(torch.sqrt(var+1e-6),dim=-1) # [N]
    
    
    mean_std_ = list(mean_std.cpu().numpy()) # [N]
    bald_ = list(mutual_information.cpu().numpy()) # [N]
    entropy_ = list(total_entropy.cpu().numpy()) # [N]
    maxsoftmax_ = list(variation_ratio.cpu().numpy()) # [N]
    
    out_eval = {'entropy_':entropy_,'maxsoftmax_':maxsoftmax_,'bald_':bald_,'mean_std_':mean_std_}
    return out_eval