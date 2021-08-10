from torchvision.datasets import MNIST,CIFAR10
from torchvision import transforms
import torch.utils.data as data
import copy
from PIL import Image

cifar10_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.49137255, 0.48235294, 0.44666667), (0.24705882, 0.24352941, 0.26156863)),
        ])

cifar10_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49137255, 0.48235294, 0.44666667), (0.24705882, 0.24352941, 0.26156863)),
        ])

class total_dataset(data.Dataset):
    def __init__(self,dataset_name='mnist',root='./dataset',train=True):
        self.dataset_name = dataset_name
        if train:
            if dataset_name == 'mnist':
                mnist = MNIST(root, download= True, train=True)
                self.x = mnist.data.unsqueeze(1).float().div(255)
                self.x = self.x.sub_(0.1307).div_(0.3081)
                self.y = mnist.targets
                self.transform = transforms.Compose([transforms.ToTensor()])
            elif dataset_name == 'cifar10':
                cifar10 = CIFAR10(root, download= True, train=True)
                self.x = cifar10.data
                self.y = cifar10.targets
                self.transform = cifar10_transform_train
        else:
            if dataset_name == 'mnist':
                mnist = MNIST(root, download= True, train=False)
                self.x = mnist.data.unsqueeze(1).float().div(255)
                self.x = self.x.sub_(0.1307).div_(0.3081)
                self.y = mnist.targets
                self.transform = transforms.Compose([transforms.ToTensor()])
            elif dataset_name == 'cifar10':
                cifar10 = CIFAR10(root, download= True, train=False)
                self.x = cifar10.data
                self.y = cifar10.targets
                self.transform = cifar10_transform_test    
    def __getitem__(self, index):
        '''
        only used for inference
        '''
        img, target = self.x[index], self.y[index]
        if self.dataset_name == 'cifar':
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return img,target
    
    def __len__(self):
        return len(self.x)

class subset_dataset(data.Dataset):
    def __init__(self,x,y,dataset_name = 'mnist'):
        self.x = x
        self.y = y
        self.dataset_name = dataset_name
        if dataset_name == 'mnist':
            self.transform = transforms.Compose([transforms.ToTensor()])
        elif dataset_name == 'cifar10':
            self.transform = cifar10_transform_train

    def __getitem__(self, index):
        '''
        only used for inference
        '''
        img, target = self.x[index], self.y[index]
        if self.dataset_name == 'cifar10':
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return img,target
        
    def __len__(self):
        return len(self.x)

class quey_dataset(data.Dataset):
    def __init__(self,x,dataset_name = 'mnist'):
        self.x = x
        self.dataset_name = dataset_name
        if dataset_name == 'mnist':
            self.transform = transforms.Compose([transforms.ToTensor()])
        elif dataset_name == 'cifar10':
            self.transform = cifar10_transform_train

    def __getitem__(self, index):
        '''
        only used for inference
        '''
        img = self.x[index]
        if self.dataset_name == 'cifar10':
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return img
        
    def __len__(self):
        return self.x.size(0)