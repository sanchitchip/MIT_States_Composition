import os 
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import pickle
import utils

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

class boolmodel(nn.Module):

    def __init__(self,dset,vargs):
        super(boolmodel,self).__init__()
        self.args = vargs
        self.dset = dset
        self.pairs = self.dset.pairs
        self.test_pairs = dset.test_pairs
        self.train_pairs = dset.train_pairs
        self.pdist_func = F.pairwise_distance

        self.in_dim = 300 if self.args.glove_init == 1 else 512
        self.mid_dim = int(np.ceil(1.5*self.in_dim))

        if self.args.m_large==1:
            self.AND_op = nn.Sequential(nn.Linear(2*self.in_dim,3*self.in_dim),
                                        nn.LeakyReLU(0.1, True),
                                        nn.Linear(3*self.in_dim,self.mid_dim),
                                        nn.LeakyReLU(0.1, True),
                                        nn.Linear(self.mid_dim,self.in_dim),
                                        nn.LeakyReLU(0.1, True))
        else:
            self.AND_op = nn.Sequential(nn.Linear(2*self.in_dim,self.mid_dim),
                                        nn.LeakyReLU(0.1, True),
                                        nn.Linear(self.mid_dim,self.in_dim),
                                        nn.LeakyReLU(0.1, True))

#            self.AND_op = nn.Sequential(nn.Linear(2*self.in_dim,self.mid_dim),
#                                        nn.LeakyReLU(0.1, True),
#                                        nn.Linear(self.mid_dim,self.in_dim))
#
            
        self.attr_embedder = nn.Embedding(len(self.dset.attrs), self.in_dim)
        self.obj_embedder = nn.Embedding(len(self.dset.objs), self.in_dim)

        if self.args.glove_init ==1:
            pretrained_weight = load_word_embeddings('/home/sanchit.chiplunkar/NAOC/MIT_States/data/glove/glove.6B.300d.txt', self.dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings('/home/sanchit.chiplunkar/NAOC/MIT_States/data/glove/glove.6B.300d.txt', self.dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)
        else:
            for idx, attr in enumerate(self.dset.attrs):
                at_id = idx
                weight = pickle.load(open('%s/svm/attr_%d'%(self.args.data_dir, at_id))).coef_.squeeze()
                self.attr_embedder.weight[idx].data.copy_(torch.from_numpy(weight))

            for idx, obj in enumerate(self.dset.objs):
                obj_id = idx
                weight = pickle.load(open('%s/svm/obj_%d'%(self.args.data_dir, obj_id))).coef_.squeeze()
                self.obj_embedder.weight[idx].data.copy_(torch.from_numpy(weight))

        for param in self.attr_embedder.parameters():
            param.requires_grad = False
        for param in self.obj_embedder.parameters():
            param.requires_grad = False

        
        if self.args.img_embed==1 or self.args.glove_init ==1:            
            self.image_feat = nn.Sequential(nn.Linear(512,self.in_dim),
                                            nn.ReLU(True))

#            self.image_feat = nn.Sequential(nn.Linear(512,self.mid_dim),
#                                             nn.LeakyReLU(0.1, True),
#                                             nn.Linear(self.mid_dim,self.in_dim),
#                                             nn.LeakyReLU(0.1, True))             
#
            
        if self.args.final_embed==1:
            self.final_embed = nn.Sequential(nn.Linear(self.in_dim,self.mid_dim),
                                             nn.LeakyReLU(0.1, True),
                                             nn.Linear(self.mid_dim,self.in_dim),
                                             nn.LeakyReLU(0.1, True))             
  
############################################################################################################
    def compose(self,attr,obj):
        vattr = self.attr_embedder(attr)
        vobj = self.obj_embedder(obj)
        vcat = torch.cat([vobj,vattr],1)
        vcomp = self.AND_op(vcat)
            
        return vcomp
############################################################################################################
    def train_forward(self, x, epoch):
        img, attr_label, obj_label = x[0], x[1], x[2]
        neg_attrs, neg_objs = x[4], x[5]
        # in this way
        vpos = self.compose(attr_label, obj_label)
        vneg = self.compose(neg_attrs, neg_objs)
        if self.args.img_embed==1 or self.args.glove_init ==1:
            img = self.image_feat(img)
        loss = F.triplet_margin_loss(img, vpos,
                                     vneg, margin=.5)
        return loss
##################################################################################################################################
    
    def val_forward(self,x):
        img, obj_label = x[0], x[2]
        batch_size = img.size(0)

        if self.args.img_embed==1 or self.args.glove_init==1:
            img = self.image_feat(img)

        attrs, objs = zip(*self.pairs)
        attrs = torch.LongTensor(attrs).cuda()
        objs = torch.LongTensor(objs).cuda()

        attr_embeds = self.compose(attrs, objs)

        dists = {}
        for i, (attr, obj) in enumerate(self.pairs):
            attr_embed = attr_embeds[i, None].expand(batch_size, attr_embeds.size(1))
            # no self.pdist_fun
            dist = self.pdist_func(img, attr_embed)
            dist = dist.unsqueeze(1)
            # before unsqueeze shape was 100 and after unsqueeze it should be [100,1]
#            print(dist.shape)
            #print(dist)
            dists[(attr, obj)] = dist.data
        attr_pred, obj_pred, score_tensor = utils.generate_prediction_tensors(dists, self.dset, obj_label.data, is_distance=True, source='manifold')

        return None, [attr_pred, obj_pred, score_tensor]
    
###################################################################################################################################

    def forward(self,x,epoch=0):
        if self.training:
            loss = self.train_forward(x,epoch)
        else:
            loss = self.val_forward(x)
        torch.cuda.empty_cache()
        return loss


class boolmodel_CE(nn.Module):

    def __init__(self,dset,vargs):
        super(boolmodel_CE,self).__init__()
        self.args = vargs
        self.dset = dset
        self.pairs = self.dset.pairs
        self.in_dim = 512
        self.mid_dim = int(np.ceil(1.5*self.in_dim))
        self.test_pairs = dset.test_pairs
        self.train_pairs = dset.train_pairs
        

        if self.args.m_large==1:
            self.AND_op = nn.Sequential(nn.Linear(2*self.in_dim,3*self.in_dim),
                                        nn.LeakyReLU(0.1, True),
                                        nn.Linear(3*self.in_dim,self.mid_dim),
                                        nn.LeakyReLU(0.1, True),
                                        nn.Linear(self.mid_dim,self.in_dim),
                                        nn.LeakyReLU(0.1, True))
        else:
            self.AND_op = nn.Sequential(nn.Linear(2*self.in_dim,self.mid_dim),
                                        nn.LeakyReLU(0.1, True),
                                        nn.Linear(self.mid_dim,self.in_dim),
                                        nn.LeakyReLU(0.1, True))


        self.attr_embedder = nn.Embedding(len(self.dset.attrs), self.in_dim)
        self.obj_embedder = nn.Embedding(len(self.dset.objs), self.in_dim)

        
        for idx, attr in enumerate(dset.attrs):
            at_id = idx
            weight = pickle.load(open('%s/svm/attr_%d'%(self.args.data_dir, at_id))).coef_.squeeze()
            self.attr_embedder.weight[idx].data.copy_(torch.from_numpy(weight))
        for idx, obj in enumerate(dset.objs):
            obj_id = idx
            weight = pickle.load(open('%s/svm/obj_%d'%(self.args.data_dir, obj_id))).coef_.squeeze()
            self.obj_embedder.weight[idx].data.copy_(torch.from_numpy(weight))

        for param in self.attr_embedder.parameters():
            param.requires_grad = False
        for param in self.obj_embedder.parameters():
            param.requires_grad = False

        

        if self.args.img_embed==1:            
            self.image_feat = nn.Sequential(nn.Linear(512,self.mid_dim),
                                            nn.LeakyReLU(0.1, True),
                                            nn.Linear(self.mid_dim,self.in_dim),
                                            nn.LeakyReLU(0.1, True))
            
        if self.args.final_embed==1:
            self.final_embed = nn.Sequential(nn.Linear(self.in_dim,self.mid_dim),
                                             nn.LeakyReLU(0.1, True),
                                             nn.Linear(self.mid_dim,self.in_dim),
                                             nn.LeakyReLU(0.1, True))             
     
  
############################################################################################################
    def compose(self,attr,obj):
        vattr = self.attr_embedder(attr)
        vobj = self.obj_embedder(obj)
        vcat = torch.cat([vobj,vattr],1)
        vcomp = self.AND_op(vcat)

        return vcomp
    
############################################################################################################
    def train_forward(self, x, epoch):
        img, attr_label, obj_label = x[0], x[1], x[2]
        if self.args.img_embed==1:
            img = self.image_feat(img)

        neg_attrs, neg_objs = x[4], x[5]
        # in this way
        vlab = np.random.binomial(1,.25,attr_label.shape[0])
        vlab = torch.from_numpy(vlab).byte().cuda()
        samp_attr,samp_obj = neg_attrs.clone(),neg_objs.clone()
        samp_attr[vlab] = attr_label[vlab]
        samp_obj[vlab] = obj_label[vlab]
        vcomp = self.compose(samp_attr,samp_obj)
        vlab = vlab.float()
        vscr = (vcomp*img).sum(1)
        vloss = nn.BCEWithLogitsLoss()
        loss = vloss(vscr,vlab)
        
        return loss
##################################################################################################################################
    
    def val_forward(self,x):
        img, obj_label = x[0], x[2]
        batch_size = img.size(0)

        if self.args.img_embed==1:
            img = self.image_feat(img)
        attrs, objs = zip(*self.pairs)
        attrs = torch.LongTensor(attrs).cuda()
        objs = torch.LongTensor(objs).cuda()

        attr_embeds = self.compose(attrs, objs)
        scores = {}
        for i, (attr, obj) in enumerate(self.pairs):
            composed_clf = attr_embeds[i, None].expand(batch_size, attr_embeds.size(1))
            ## observe the score values
            score = torch.sigmoid((img*composed_clf).sum(1)).unsqueeze(1)
            # shape is [100[bs],1]
            scores[(attr, obj)] = score.data
        attr_pred, obj_pred, score_tensor = utils.generate_prediction_tensors(scores, self.dset, obj_label.data, is_distance=False, source='manifold')


        return None, [attr_pred, obj_pred, score_tensor]
    
###################################################################################################################################

    def forward(self,x,epoch=0):
        if self.training:
            loss = self.train_forward(x,epoch)
        else:
            loss = self.val_forward(x)
        torch.cuda.empty_cache()
        return loss



