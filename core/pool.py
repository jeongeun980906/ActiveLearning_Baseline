from numpy import dtype
from core.data import total_dataset,subset_dataset,quey_dataset
import torch
import copy
import random

class AL_pool():
    def __init__(self,root='./dataset',dataset_name='mnist',num_init=100):
        self.basedata=total_dataset(dataset_name, root=root)
        self.batch_size=128
        self.idx = torch.tensor(random.sample(range(self.basedata.__len__()), num_init))
    
    def subset_dataset(self,indices):
        indices = torch.cat((self.idx,indices),0)
        self.idx = indices
        x = copy.deepcopy(self.basedata.x[indices])
        y = copy.deepcopy(self.basedata.y[indices])
        total = torch.range(0,self.basedata.__len__()-1,dtype=torch.int64)
        mask = torch.ones(total.numel(), dtype=torch.bool)
        mask[self.idx] = False
        self.unlabled_idx = total[mask]
        labeled_subset = subset_dataset(x,y)
        train_loader = torch.utils.data.DataLoader(labeled_subset, batch_size=self.batch_size, 
                        shuffle=False)
        infer_loader = self.get_unlabled_pool()
        return train_loader,infer_loader

    def get_unlabled_pool(self):
        print(self.unlabled_idx.size())
        x = copy.deepcopy(self.basedata.x[self.unlabled_idx])
        query_pool = quey_dataset(x)
        loader  = torch.utils.data.DataLoader(query_pool, batch_size=self.batch_size, 
                        shuffle=False)
        return loader

if __name__ == '__main__':
    p = AL_pool()
    _,_ = p.subset_dataset(torch.zeros(size=(0,1),dtype=torch.int64).squeeze(1))
    _,_ = p.subset_dataset(torch.LongTensor([1,2,3]))