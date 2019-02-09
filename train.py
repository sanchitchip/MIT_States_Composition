import torch
import numpy as np
import boolmodel
import model_demorgan
import torch.optim as optim
import argparse
from sklearn import metrics
import sys
import os
import logging
import pickle
import dataset as dset
import utils
import NAOC_bool

############################################################
def train(vepoch,vargs):
    vbatch = vargs.batch_size
    # TO-DO use proper dataloading methodology for training of this thing.
    vmodel.train()
    vloss = []
    for idx,data in enumerate(trainloader):
        optimizer.zero_grad()
        data = [d.to(vdevice) for d in data]
        loss = vmodel(data,vepoch)
        loss.backward()
        optimizer.step()
        vloss.append(float(loss))
    vloss = sum(vloss)/len(vloss)
    log.info("Loss at epoch {} is {}".format(vepoch,vloss))
    return vloss

################################################
def test(epoch):

    vmodel.eval()

    accuracies = []
    log.info("test Epoch is %d" %(epoch))
    for idx, data in enumerate(testloader):
        
        data = [d.to(vdevice) for d in data]
#        _, [attr_pred, obj_pred, _] = vmodel(data,(attr_emb,obj_emb))
        _, [attr_pred, obj_pred, _] = vmodel(data)
        # shape of attr_pred is [bs,3]:: 3 is open,closed nd orcl attr_id
        match_stats = utils.performance_stats(attr_pred, obj_pred, data)
        accuracies.append(match_stats)

    accuracies = zip(*accuracies)
    accuracies = map(torch.mean, map(torch.cat, accuracies))
    attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc = accuracies
    print( '(test) E: %d | A: %.3f | O: %.3f | Cl: %.3f | Op: %.4f | OrO: %.4f'%(epoch, attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc))
######################################################################


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str,help='data root dir for panda merging etc..')
parser.add_argument('--batch_size', type=int, default=100, help='batch size for iterations.')
parser.add_argument('--epochs', type=int, default=1500, help='number of epochs.')    
parser.add_argument('--model', default='boolmodel', help='boolmodel|boolmodel_CE|NAOC_bool')
parser.add_argument('--img_embed', type=int, default=0, help='whether to add MLP after VGG img feature.')
parser.add_argument('--final_embed', type=int, default=0, help='whether to add MLP after final expression computation.')
parser.add_argument('--eval_every', type=int, default=25, help='after what interval should model be evaluated.')
parser.add_argument('--not_frozen', type=int, default=0, help='whether to freeze the modules.')
parser.add_argument('--elmo', type=int, default=0, help='whether to use the elmo embedding.')
parser.add_argument('--glove_init', type=int, default=0, help='whether to use the glove  embedding.')

parser.add_argument('--model_load', type=str,default = "/home/sanchit.chiplunkar/NAOC/MIT_States/model_save/svm_saved_model_ep_5000_500.pth",help='path to where model is saved')

parser.add_argument('--wt_load', type=int, default=0, help='whether to load the models from model_load argument.')
parser.add_argument('--m_large', type=int, default=0, help='whether to load the 3 layered MLP.')

parser.add_argument('--adaptive_lr', type=int, default=0, help='whether to use the adaptive learning rate.')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay to use.')
parser.add_argument('--lr', type=float, default=0.01, help='lr value to use.')
parser.add_argument('--optim', type=str, default="Adam", help='which optimizer to use.')



logging.basicConfig()
log = logging.getLogger()

log.setLevel(logging.INFO)

args = parser.parse_args()
epoch = args.epochs
log.info("showing args value")
log.info(args)


log.info("is cuda activated:: {}".format(torch.cuda.is_available()))

split = 'compositional-split'
DSet = dset.MITStatesActivations
trainset = DSet(root=args.data_dir, phase='train', split=split)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
testset = DSet(root=args.data_dir, phase='test', split=split)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)


if torch.cuda.is_available():
    vdevice = torch.device("cuda")
else:
    vdevice = torch.device("cpu")

if args.model=="boolmodel":
    vmodel = boolmodel.boolmodel(trainset,args)
elif args.model=="boolmodel_CE":
    vmodel = boolmodel.boolmodel_CE(trainset,args)
elif args.model=="NAOC_bool":
    vmodel = NAOC_bool.NAOC_bool(trainset,args)
else:
    log.warning("no other model for this folder yet.")

if args.wt_load == 1:
    if args.m_large ==1:
        vmod = model_demorgan.demorgan_large(args)        
    else:            
        vmod = model_demorgan.demorgan(args)
    vload = args.model_load
    vmod.load_state_dict(torch.load(vload,map_location="cpu"))
    log.info("loaded the model.")
    log.info("updating the AND_op weights")
    vmodel.AND_op.load_state_dict(vmod.AND_op.state_dict())
    del vmod
else:
    log.info("training on random init of AND operator similar to RedWine")

vmodel = vmodel.to(vdevice)
log.info("model loaded on cuda")
if args.not_frozen == 0:    
    for p in vmodel.AND_op.parameters():
        p.requires_grad = False


 
if args.img_embed==1 or args.glove_init == 1:
    params = list(vmodel.image_feat.parameters())
    if args.not_frozen==1:
        params = params  + list(vmodel.AND_op.parameters())
elif args.img_embed==1 and args.final_embed==1:
    params = list(vmodel.image_feat.parameters()) + list(vmodel.final_embed.parameters())
    if args.not_frozen==1:
        params = params + list(vmodel.AND_op.parameters()) 

elif args.final_embed==1:    
    params = list(vmodel.final_embed.parameters())
    if args.not_frozen==1:
        params = params + list(vmodel.AND_op.parameters())
else:
    params = vmodel.parameters()




params = filter(lambda p: p.requires_grad, params)
wd = args.wd
vlr = args.lr
# try with SGD optimizer.
optimizer = optim.Adam(params, lr = vlr,weight_decay=wd)    

if args.optim == "SGD":
    optimizer = optim.SGD(params, lr=vlr, weight_decay=wd, momentum=0.9)


if args.adaptive_lr==1:
#    sched = optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=.1)
    sched  = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 'min',
                                                  factor=.1,
                                                 patience=10,
                                                 verbose=True)

for i in range(epoch):
    log.info("epoch is {}".format(i))        

    vloss=train(i,args)
    torch.cuda.empty_cache()

    if args.adaptive_lr==1:
        sched.step(vloss)
    #TO-DO add argument for test_every_epoch in parser.
    if i%args.eval_every==0:
        test(i)
    
