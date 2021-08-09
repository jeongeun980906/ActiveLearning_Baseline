import os
import json

def print_n_txt(_f,_chars,_addNewLine=True,_DO_PRINT=True):
    if _addNewLine: _f.write(_chars+'\n')
    else: _f.write(_chars)
    _f.flush();os.fsync(_f.fileno()) # Write to txt
    if _DO_PRINT:
        print (_chars)

class Logger():
    def __init__(self,path,init_indicies):
        self.path = path
        self.train_acc = []
        self.test_acc = []
        self.idx = {}
        self.idx[0]=init_indicies.numpy().tolist()
        self.flag=1
    
    def append(self,train_acc,test_acc,new):
        self.train_acc.append(train_acc)
        self.test_acc.append(test_acc)
        self.idx[self.flag]=new.numpy().tolist()
        self.flag+=1
        
    def save(self):
        data = {}
        with open(self.path,'w') as json_file:
            data['train_accr']=self.train_acc
            data['test_accr']=self.test_acc
            data['pool index']= self.idx
            json.dump(data,json_file, indent=4)