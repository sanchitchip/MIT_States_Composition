import torch
import numpy as np
import pickle
import logging
import model_demorgan
import torch.optim as optim
import argparse
import torch.nn as nn
import dataset as dset


def train(epoch):
    vloss =[]
    if epoch<=args.el_epochs:
        vfinal = torch.cat((attr_emb,obj_emb))
        vlist = np.arange(vfinal.shape[0])
        np.random.shuffle(vlist)
        vbatch = 30
        for i in range(0,int(np.ceil(len(vlist)/vbatch))):
            model.train()
            optimizer.zero_grad()
            vstrt = i*vbatch
            vend = (i+1)*vbatch
            if i == int(np.ceil(len(vlist)/vbatch))-1:
                vend = len(vlist)
            loss = model(vfinal[vlist[vstrt:vend]],epoch)
            loss.backward()
            optimizer.step()
            vloss.append(float(loss))
    else:
        vbatch  = 200
        vtrain_pairs = list(zip(*trainset.train_pairs))        
        for idx in range(0,int(np.ceil(len(vtrain_pairs[0])/vbatch))):   
            model.train()
            optimizer.zero_grad()
            vstrt = idx*vbatch
            vend = (idx+1)*vbatch            
            if idx== int(np.ceil(len(vtrain_pairs[0])/vbatch))-1:
                vend = len(vtrain_pairs[0])
                
            vattr_idx = torch.LongTensor(vtrain_pairs[0][vstrt:vend]).cuda()
            vobj_idx = torch.LongTensor(vtrain_pairs[1][vstrt:vend]).cuda()
            vattr = attr_embedder(vattr_idx)
            vobj = obj_embedder(vobj_idx)
            loss = model((vattr,vobj),epoch)
            loss.backward()
            optimizer.step()
            vloss.append(float(loss))

    vloss = sum(vloss)/len(vloss)
    log.info("loss value after epoch {} is {}".format(epoch,vloss))
    return vloss
#################################################################################################
def load_word_embeddings(emb_file, vocab):

    vocab = [v.lower() for v in vocab]

    embeds = {}
    for line in open(emb_file, 'r'):
        line = line.strip().split(' ')
        wvec = torch.Tensor(map(float, line[1:]))
        embeds[line[0]] = wvec

    # for zappos (should account for everything)
    custom_map = {'Faux.Fur':'fur', 'Faux.Leather':'leather', 'Full.grain.leather':'leather', 'Hair.Calf':'hair', 'Patent.Leather':'leather', 'Nubuck':'leather', 
                'Boots_Ankle':'boots', 'Boots_Knee_High':'knee-high', 'Boots_Mid-Calf':'midcalf', 'Shoes_Boat_Shoes':'shoes', 'Shoes_Clogs_and_Mules':'clogs',
                'Shoes_Flats':'flats', 'Shoes_Heels':'heels', 'Shoes_Loafers':'loafers', 'Shoes_Oxfords':'oxfords', 'Shoes_Sneakers_and_Athletic_Shoes':'sneakers'}
    for k in custom_map:
        embeds[k.lower()] = embeds[custom_map[k]]

    embeds = [embeds[k] for k in vocab]
    embeds = torch.stack(embeds)
    print 'loaded embeddings', embeds.size()

    return embeds



#################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='data directory')

parser.add_argument('--CUDA', type=int,default=0, help='train/test with cuda or not.')
parser.add_argument('--batch_size', type=int, default=100, help='batch size for iterations.')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs.')    
parser.add_argument('--el_epochs', type=int, default=200, help='number of epochs for training elemental objects.')    
parser.add_argument('--model_save',type=str, help='path where model is to be saved')
parser.add_argument('--model',type=str, default = "demorgan", help='demorgan|demorgan_large')

parser.add_argument('--elmo', type=int,default=0, help='whether to use elmo embedding or svm prims')
parser.add_argument('--glove_init', type=int,default=0, help='whether to use glove embedding or svm prims')
parser.add_argument('--optim', type=str,default="Adam", help='which optimizer to use')


parser.add_argument('--eval_every', type=int,default=100, help='to run eval every what epoch?')
parser.add_argument('--save_every', type=int,default=1000, help='to save every what epoch?')
parser.add_argument('--adaptive_lr', type=int,default=0, help='whether to use adaptive lr')

parser.add_argument('--lr', type=float,default=.001, help='lr')
parser.add_argument('--wd', type=float,default=1e-4, help='weight decay')

#################################################################################################

logging.basicConfig()
log = logging.getLogger()             

log.setLevel(logging.INFO)
args = parser.parse_args()

vdata_dir = "/home/sanchit.chiplunkar/NAOC/MIT_States/data"

if torch.cuda.is_available():
    vdevice = torch.device("cuda")
else:
    vdevice = torch.device("cpu")

DSet = dset.MITStatesActivations
split = 'compositional-split'
trainset = DSet(root=args.data_dir, phase='train', split=split)
in_dim = 300 if args.glove_init == 1 else 512 

attr_embedder = nn.Embedding(len(trainset.attrs), in_dim)
obj_embedder = nn.Embedding(len(trainset.objs), in_dim)




if args.glove_init==1:
    log.info("Loading the glove embeddings for Attributes and objects")
    pretrained_weight = load_word_embeddings('/home/sanchit.chiplunkar/NAOC/MIT_States/data/glove/glove.6B.300d.txt', trainset.attrs)
    attr_embedder.weight.data.copy_(pretrained_weight)
    pretrained_weight = load_word_embeddings('/home/sanchit.chiplunkar/NAOC/MIT_States/data/glove/glove.6B.300d.txt', trainset.objs)
    obj_embedder.weight.data.copy_(pretrained_weight)
    log.info("Loaded the glove embeddings for Attributes and objects")

else:
    log.info("Loading the SVM embeddings for Attributes")
    for idx, attr in enumerate(trainset.attrs):
        at_id = idx
        weight = pickle.load(open('%s/svm/attr_%d'%(vdata_dir, at_id))).coef_.squeeze()
        attr_embedder.weight[idx].data.copy_(torch.from_numpy(weight))

    log.info("Loading the SVM embeddings for Object")
    for idx, obj in enumerate(trainset.objs):
        obj_id = idx
        weight = pickle.load(open('%s/svm/obj_%d'%(vdata_dir, obj_id))).coef_.squeeze()
        obj_embedder.weight[idx].data.copy_(torch.from_numpy(weight))

for param in attr_embedder.parameters():
    param.requires_grad = False
for param in obj_embedder.parameters():
    param.requires_grad = False

vattr= 115
vobj = 245

vattr_ind = torch.LongTensor(np.arange(vattr))
attr_emb = torch.Tensor(attr_embedder(vattr_ind))
attr_emb = attr_emb.to(vdevice)
log.info(attr_emb.is_cuda)
vobj_ind = torch.LongTensor(np.arange(vobj))
obj_emb = torch.Tensor(obj_embedder(vobj_ind))
obj_emb = obj_emb.to(vdevice)

log.info(obj_emb.is_cuda)

attr_embedder = attr_embedder.to(vdevice)
obj_embedder = obj_embedder.to(vdevice)
log.info("Loaded the embeddings for Attributes and Object to GPU ")

log.info("CHECK created tensor for attr and obj")
log.info(attr_emb.is_cuda)
log.info(obj_emb.is_cuda)
log.info("CHECK DONE")

log.info("Loading the model")
if args.model == "demorgan":
    model = model_demorgan.demorgan(args)
    model = model.to(vdevice)
elif args.model == "demorgan_large":
    model = model_demorgan.demorgan_large(args)
    model = model.to(vdevice)
log.info("model loaded and transfered to cuda")

log.info("defining params and optimizer")
params = filter(lambda p:p.requires_grad, model.parameters())
optimizer = optim.Adam(params,lr=args.lr,weight_decay = args.wd)

if args.optim == "SGD":
    log.info("using the SGD as optimizer")
    optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.wd, momentum=0.9)

    
if args.adaptive_lr==1:
    log.info("defining the schdeules for optimizer")
    sched  = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                  'min',
                                                  factor=.1,
                                                  patience=10,
                                                  verbose=True)

vPath = args.model_save
vepoch = args.epochs
for i in range(vepoch):
    vloss = train(i)
    sched.step(vloss)
    if i%args.save_every==0:
        torch.save(model.state_dict(),vPath)
        log.info("model saved at epoch {}".format(i))



#    else:
##### SOME MAJOR BUG HERE
#        vbatch = 30
#        vbig = np.arange(obj_emb.shape[0])
#        np.random.shuffle(vbig)
#        vshort = np.arange(attr_emb.shape[0])
#        for i in range(0,int(np.ceil(len(vbig)/vbatch))):
#            model.train()
#            vstrt = i*vbatch
#            vend = (i+1)*vbatch
#            if i == int(np.ceil(len(vbig)/vbatch))-1:
#                vend = len(vbig)
#
#            vrand = np.random.choice(vshort,vend-vstrt,replace=False)
#            optimizer.zero_grad()
#            loss = model((obj_emb[vbig[vstrt:vend]],
#                          attr_emb[vshort[vrand]]),epoch)
#            loss.backward()
#            optimizer.step()
#            vloss.append(float(loss))
