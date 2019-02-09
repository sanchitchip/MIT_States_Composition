import os 
import pdb
import numpy as np
import torch
import torch.nn as nn
import h5py
import pickle


class demorgan(nn.Module):
    def __init__(self,args):
        super(demorgan,self).__init__()
        self.args=args
        self.in_dim = 300 if self.args.glove_init ==1 else 512 
        self.mid_dim = int(np.ceil(1.5*self.in_dim))        

        self.AND_op = nn.Sequential(nn.Linear(2*self.in_dim,self.mid_dim),
                                    nn.LeakyReLU(0.1, True),
                                    nn.Linear(self.mid_dim,self.in_dim),
                                    nn.LeakyReLU(0.1, True))

        self.OR_op = nn.Sequential(nn.Linear(2*self.in_dim,self.mid_dim),
                                   nn.LeakyReLU(0.1, True),
                                   nn.Linear(self.mid_dim,self.in_dim),
                                   nn.LeakyReLU(0.1, True))
        
        self.NOT_op = nn.Sequential(nn.Linear(self.in_dim,self.mid_dim),
                                   nn.LeakyReLU(0.1, True),
                                   nn.Linear(self.mid_dim,self.in_dim),
                                   nn.LeakyReLU(0.1, True))

        #########################END OF OPERATORS ###############################
        
        self.ops_dic = dict()                
        self.ops_dic["NOT"] = lambda v1: self.NOT_op(v1)
        self.ops_dic["AND"] = lambda v1,v2: self.AND_op(torch.cat([v1,v2]))
        self.ops_dic["OR"] = lambda v1,v2: self.OR_op(torch.cat([v1,v2]))
## AND :::: A' OR B' OR :::: A' AND B'
        self.ops_par = dict()
        self.ops_par["NOT"] = lambda v1: self.NOT_op(v1)
        self.ops_par["AND"] = lambda v1,v2: self.OR_op(torch.cat([self.NOT_op(v1),self.NOT_op(v2)]))
        self.ops_par["OR"] = lambda v1,v2: self.AND_op(torch.cat([self.NOT_op(v1),self.NOT_op(v2)]))


        
    def el_forward(self,x):
        side1,side2 = [],[]
        vbatch = x.shape[0]
        
        for i in range(vbatch):            
            side1.append(x[i])
            side2.append(self.NOT_op(self.NOT_op(x[i])))

            side1.append(x[i])
            side2.append(self.AND_op(torch.cat((x[i],x[i]))))

            side1.append(x[i])
            side2.append(self.OR_op(torch.cat((x[i],x[i]))))
        side1,side2 = torch.stack(side1),torch.stack(side2)
        loss = nn.MSELoss()
        vloss = loss(side1,side2)
        return vloss
    
    def comp_forward(self,x):
# 3 eqn::: A AND B == B AND A || (A AND B)' == A' or B' ||
# (A OR B)' == A' AND B'

        vobj,vattr = x
        side1,side2 = [],[]
        vbatch = vobj.shape[0]
        for i in range(vbatch):
            vAND = self.ops_dic["AND"](vobj[i],vattr[i])
            vOR = self.ops_dic["OR"](vobj[i],vattr[i])
            side1.append(vAND)
            side2.append(self.ops_dic["AND"](vattr[i],vobj[i]))

            side1.append(self.NOT_op(vAND))
            side2.append(self.ops_par["AND"](vobj[i],vattr[i]))

            side1.append(self.NOT_op(vOR))
            side2.append(self.ops_par["OR"](vobj[i],vattr[i]))


        side1,side2 = torch.stack(side1),torch.stack(side2)
        loss = nn.MSELoss()
        vloss = loss(side1,side2)
        return vloss
    
    def forward(self,x,epoch):
        if epoch<=self.args.el_epochs:
            vloss = self.el_forward(x)
        else:
            vloss = self.comp_forward(x)
        return vloss
    
    

## MODE for 3 layered MLP
class demorgan_large(nn.Module):
    def __init__(self,args):
        super(demorgan_large,self).__init__()
        self.args=args
        self.in_dim = 300 if self.args.glove_init ==1 else 512
        self.mid_dim = int(np.ceil(1.5*self.in_dim))        

        self.AND_op = nn.Sequential(nn.Linear(2*self.in_dim,3*self.in_dim),
                                    nn.LeakyReLU(0.1, True),
                                    nn.Linear(3*self.in_dim,self.mid_dim),
                                    nn.LeakyReLU(0.1, True),
                                    nn.Linear(self.mid_dim,self.in_dim),
                                    nn.LeakyReLU(0.1, True))

        self.OR_op = nn.Sequential(nn.Linear(2*self.in_dim,3*self.in_dim),
                                    nn.LeakyReLU(0.1, True),
                                    nn.Linear(3*self.in_dim,self.mid_dim),
                                    nn.LeakyReLU(0.1, True),
                                    nn.Linear(self.mid_dim,self.in_dim),
                                    nn.LeakyReLU(0.1, True))

        
        self.NOT_op = nn.Sequential(nn.Linear(self.in_dim,self.mid_dim),
                                   nn.LeakyReLU(0.1, True),
                                   nn.Linear(self.mid_dim,self.in_dim),
                                   nn.LeakyReLU(0.1, True))

        #########################END OF OPERATORS ###############################
        
        self.ops_dic = dict()                
        self.ops_dic["NOT"] = lambda v1: self.NOT_op(v1)
        self.ops_dic["AND"] = lambda v1,v2: self.AND_op(torch.cat([v1,v2]))
        self.ops_dic["OR"] = lambda v1,v2: self.OR_op(torch.cat([v1,v2]))
## AND :::: A' OR B' OR :::: A' AND B'
        self.ops_par = dict()
        self.ops_par["NOT"] = lambda v1: self.NOT_op(v1)
        self.ops_par["AND"] = lambda v1,v2: self.OR_op(torch.cat([self.NOT_op(v1),self.NOT_op(v2)]))
        self.ops_par["OR"] = lambda v1,v2: self.AND_op(torch.cat([self.NOT_op(v1),self.NOT_op(v2)]))


        
    def el_forward(self,x):
        side1,side2 = [],[]
        vbatch = x.shape[0]
        
        for i in range(vbatch):            
            side1.append(x[i])
            side2.append(self.NOT_op(self.NOT_op(x[i])))

            side1.append(x[i])
            side2.append(self.AND_op(torch.cat((x[i],x[i]))))

            side1.append(x[i])
            side2.append(self.OR_op(torch.cat((x[i],x[i]))))
        side1,side2 = torch.stack(side1),torch.stack(side2)
        loss = nn.MSELoss()
        vloss = loss(side1,side2)
        return vloss
    
    def comp_forward(self,x):
# 3 eqn::: A AND B == B AND A || (A AND B)' == A' or B' ||
# (A OR B)' == A' AND B'

        vobj,vattr = x
        side1,side2 = [],[]
        vbatch = vobj.shape[0]
        for i in range(vbatch):
            vAND = self.ops_dic["AND"](vobj[i],vattr[i])
            vOR = self.ops_dic["OR"](vobj[i],vattr[i])
            side1.append(vAND)
            side2.append(self.ops_dic["AND"](vattr[i],vobj[i]))

            side1.append(self.NOT_op(vAND))
            side2.append(self.ops_par["AND"](vobj[i],vattr[i]))

            side1.append(self.NOT_op(vOR))
            side2.append(self.ops_par["OR"](vobj[i],vattr[i]))


        side1,side2 = torch.stack(side1),torch.stack(side2)
        loss = nn.MSELoss()
        vloss = loss(side1,side2)
        return vloss
    
    def forward(self,x,epoch):
        if epoch<=self.args.el_epochs:
            vloss = self.el_forward(x)
        else:
            vloss = self.comp_forward(x)
        return vloss
