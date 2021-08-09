from core.pool import AL_pool
from core.solver import solver
from core.data import total_dataset
import torch

p = AL_pool()
test_dataset = total_dataset(train=False)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=128, 
                        shuffle=False)
AL_solver = solver()
AL_solver.init_param()

train_iter,query_iter = p.subset_dataset(torch.zeros(size=(0,1),dtype=torch.int64).squeeze(1))

for i in range(20):
    AL_solver.train_image(train_iter)
    AL_solver.test_acc(test_iter)
    id = AL_solver.query_data(query_iter)
    new = p.unlabled_idx[id]
    train_iter,query_iter = p.subset_dataset(new)