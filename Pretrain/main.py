
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pk
import numpy as np
from utils import *
from tqdm import tqdm
from signal_data_add import get_signal
from torch.utils.data import  DataLoader
import pywt
import matplotlib.pyplot as plt
from patch_encoder_and_projection_head_and_classifier import *
from pytorch_pretrained_vit import ViT
import os
from models import *
import pandas as pd


def pretrain(model,projection_head,epoches,converter,optimizer,data_loader,save_path):
    for i in range(epoches):
        with tqdm(total=len(data_loader)) as p_bar:
            for batch_idx, (singals, CWTs, labels) in enumerate(data_loader):
                singals = singals.cuda()
                videos = converter(singals)
                CWTs = CWTs.cuda()
                labels = labels.cuda()

                singals_feature,CWTs_feature,videos_feature = model(singals,CWTs,videos)
                singals_projection,CWTs_projection,videos_projection = projection_head(singals_feature,CWTs_feature,videos_feature)
                                                                                                        
                loss1 = compute_loss(singals_projection,videos_projection)#去噪，兼顾信息增益（projection head不同）
                loss2 = compute_loss(singals_projection,CWTs_projection)#去噪，兼顾信息增益（projection head不同）
                loss4 = compute_loss(videos_projection,CWTs_projection)
                loss = loss1 + loss2 + loss4 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                p_bar.set_description("pretrain: epoch:{} batch_idx:{} loss:{:02f}".format(i,batch_idx,loss))
                p_bar.update()

    torch.save(model.state_dict(),os.path.join(save_path,'model_state.pth'))
    torch.save(projection_head.state_dict(),os.path.join(save_path,'projection_head_state.pth'))
        
def classifier_train(model,classifier,epoches,converter,optimizer,train_loader,val_loader,save_path):
    max_acc = 0
    for epoch in range(epoches):
        with tqdm(total=len(train_loader)) as p_bar:
            for batch_idx, (singals, CWTs, labels) in enumerate(train_loader):
                singals = singals.cuda()
                videos = converter(singals)
                CWTs = CWTs.cuda()
                labels = labels.cuda()
                singals_feature,CWTs_feature,videos_feature = model(singals,CWTs,videos)
                feature = torch.cat([singals_feature,CWTs_feature,videos_feature],dim = -1)
                pre = classifier(feature)
                loss = F.cross_entropy(pre,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                p_bar.set_description("classifier_train: epoch:{} batch_idx:{} loss:{:02f}".format(epoch, batch_idx, loss))
                p_bar.update()
        if epoch % 4 == 0 and epoch != 0:
            acc = test(model,classifier,val_loader)
            if acc>max_acc:
                print("max_acc:",acc)
                torch.save(model.state_dict(),os.path.join(save_path,'model_state.pth'))
                torch.save(classifier.state_dict(),os.path.join(save_path,'classifier_state.pth'))
                max_acc = acc

    acc = test(model,classifier,val_loader)
    if acc>max_acc:
        print("max_acc:",acc)
        torch.save(model.state_dict(),os.path.join(save_path,'model_state.pth'))
        torch.save(classifier.state_dict(),os.path.join(save_path,'classifier_state.pth'))
        max_acc = acc



def test(model,classifier,data_loader,snrs = None,snr_indexs = None,save_path = None,final = False):
    prediction = []
    true = []
    if final:
        model.load_state_dict(torch.load(os.path.join(save_path,'model_state.pth')))
        classifier.load_state_dict(torch.load(os.path.join(save_path,'classifier_state.pth')))
    for batch_idx, (singals, CWTs, labels) in enumerate(data_loader):
        singals = singals.cuda()
        videos = converter(singals)
        CWTs = CWTs.cuda()
        labels = labels.cuda()
        singals_feature,CWTs_feature,videos_feature = model(singals,CWTs,videos)
        feature = torch.cat([singals_feature,CWTs_feature,videos_feature],dim = -1)
        result = torch.argmax(classifier(feature),dim = 1)
        prediction.extend(result.cpu().numpy())
        true.extend(labels.cpu().numpy())
    if not final:
        return sum(np.array(prediction) == np.array(true))/len(true)
    else:
        prediction = np.array(prediction)
        true = np.array(true)
        acc = {}
        for i in range(len(snrs)):
            true_label = true[snr_indexs[i]]
            #print(true_label.shape)
            pre_label = prediction[snr_indexs[i]]
            cor = np.sum(true_label == pre_label)
            acc[snrs[i]] = 1.0 * cor / true_label.shape[0]
        total_acc = sum(np.array(prediction) == np.array(true))/len(true)
        acc['total'] = total_acc
        ACC = pd.DataFrame(acc.items())
        ACC.to_csv(os.path.join(save_path,'result.csv'))
        return total_acc



bsz = 100
traindataset,valset,testset,classes,snrs,snr_indexs = get_signal('/root/autodl-tmp/RML2016.10a_dict.pkl','/root/autodl-tmp/CWTdata.pkl',1,L=1,snrs_index=0)
ctraindataset,cvalset,ctestset,cclasses,csnrs,csnr_indexs = get_signal('/root/autodl-tmp/RML2016.10a_dict.pkl','/root/autodl-tmp/CWTdata.pkl',0.3,L=1,snrs_index=0)


converter = video_Converter(bsz,14)
train_loader = torch.utils.data.DataLoader(traindataset, batch_size=bsz, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=bsz, shuffle=False)
test_loader = torch.utils.data.DataLoader(testset, batch_size=bsz, shuffle=False)

ctrain_loader = torch.utils.data.DataLoader(ctraindataset, batch_size=bsz, shuffle=True)
cval_loader = torch.utils.data.DataLoader(cvalset, batch_size=bsz, shuffle=False)
ctest_loader = torch.utils.data.DataLoader(ctestset, batch_size=bsz, shuffle=False)

root_path = '/root/autodl-tmp/不同标签量实验/不微调'



for epoch in [5,10,15,20,50,100]:
    model = Model(160,16,2).cuda()
    projection_head = Proj(160,160).cuda()#160或者128
    model.train()
    projection_head.train()
    classifier = classifier_head(3*160,11).cuda()
    classifier.train()

    save_path = os.path.join(root_path,str(epoch))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    optimizer = torch.optim.Adam(list(model.parameters())+list(projection_head.parameters()),lr = 0.0005)    
    pretrain(model,projection_head,epoch,converter,optimizer,train_loader,save_path)
    optimizer = torch.optim.Adam(classifier.parameters(), lr = 0.001)
    classifier_train(model,classifier,20,converter,optimizer,ctrain_loader,cval_loader,save_path)
    test(model,classifier,ctest_loader,csnrs,csnr_indexs,save_path,final = True)







