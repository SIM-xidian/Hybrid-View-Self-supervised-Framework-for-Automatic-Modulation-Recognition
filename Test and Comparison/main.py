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
import time

from fvcore.nn import FlopCountAnalysis

def train(model_name,classifier,epoches,train_loader,val_loader,save_path,model=None,finetunning = False,converter = None):

    if model_name == 'HVSF' or model_name == 'HVSF_wo_finetunning':
        if finetunning:
            optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.0003},{'params': classifier.parameters(), 'lr': 0.001}])
        else:
            optimizer = torch.optim.Adam(classifier.parameters(), lr= 0.001)
            
        max_acc = 0
        for epoch in range(epoches):
            time1 = time.time()
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
            time2 = time.time()
            if epoch % 10 == 0 and epoch != 0:
                acc = test(model_name,classifier,val_loader,model = model,converter = converter)
                if acc>max_acc:
                    print("max_acc:",acc)
                    torch.save(model.state_dict(),os.path.join(save_path,'model_state.pth'))
                    torch.save(classifier.state_dict(),os.path.join(save_path,'classifier_state.pth'))
                    max_acc = acc
        allocated_memory = torch.cuda.memory_allocated() / 1024 ** 2  # MB
        cached_memory = torch.cuda.memory_reserved() / 1024 ** 2  # MB
        acc = test(model_name,classifier,val_loader,model = model,converter = converter)
        if acc>max_acc:
            print("max_acc:",acc)
            torch.save(model.state_dict(),os.path.join(save_path,'model_state.pth'))
            torch.save(classifier.state_dict(),os.path.join(save_path,'classifier_state.pth'))
            max_acc = acc
        epoch_time = time2-time1

        singals, CWTs, labels = next(iter(train_loader))
        singals = singals[0:1]
        CWTs = CWTs[0:1]
        singals = singals.cuda()
        CWTs = CWTs.cuda()
        converter_tmp = video_Converter(1,14)
        videos = converter_tmp(singals)
        per_sample_start_time = time.time()
        singals_feature,CWTs_feature,videos_feature = model(singals,CWTs,videos)
        feature = torch.cat([singals_feature,CWTs_feature,videos_feature],dim = -1)
        pre = classifier(feature)
        per_sample_end_time = time.time()
        infer_one_sample_time = per_sample_end_time - per_sample_start_time
        
        flops1 = FlopCountAnalysis(model,(singals,CWTs,videos))
        flops2 = FlopCountAnalysis(classifier, feature)
        flops = flops1.total() + flops2.total()

        num_params1 = sum(p.numel() for p in model.parameters())
        num_params2 = sum(p.numel() for p in classifier.parameters()) 
        num_params = num_params1+num_params2
        
        info = {}
        info['train_epoch_time(bsz = 100sample)'] = epoch_time
        info['infer_one_sample_time'] = infer_one_sample_time
        info['flops(infer_1_sample)'] = flops
        info['num_params(infer)'] = num_params
        info['allocated_memory'] = allocated_memory
        info['cached_memory'] = cached_memory
        info = pd.DataFrame(info.items())
        info.to_csv(os.path.join(save_path,'info.csv'))



    else:
        optimizer = torch.optim.Adam(classifier.parameters(), lr = 0.001)
        max_acc = 0
        for epoch in range(epoches):
            time1 = time.time()
            with tqdm(total=len(train_loader)) as p_bar:
                for batch_idx, (data,_, labels) in enumerate(train_loader):
                    data = data.cuda()
                    labels = labels.cuda()
                    pre = classifier(data)
                    loss = F.cross_entropy(pre,labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    p_bar.set_description("classifier_train: epoch:{} batch_idx:{} loss:{:02f}".format(epoch, batch_idx, loss))
                    p_bar.update()
            time2 = time.time()
            if epoch % 100 == 0 and epoch != 0:
                acc = test(model_name,classifier,val_loader)
                if acc>max_acc:
                    print("max_acc:",acc)
                    torch.save(classifier.state_dict(),os.path.join(save_path,'classifier_state.pth'))
                    max_acc = acc
        allocated_memory = torch.cuda.memory_allocated() / 1024 ** 2  # MB
        cached_memory = torch.cuda.memory_reserved() / 1024 ** 2  # MB
        acc = test(model_name,classifier,val_loader)
        if acc>max_acc:
            print("max_acc:",acc)
            torch.save(classifier.state_dict(),os.path.join(save_path,'classifier_state.pth'))
            max_acc = acc

        epoch_time = time2-time1

        data,_, labels = next(iter(train_loader))
        data = data[0:1]
        data = data.cuda()
        per_sample_start_time = time.time()
        pre = classifier(data)
        per_sample_end_time = time.time()
        infer_one_sample_time = per_sample_end_time - per_sample_start_time
        
        flops = FlopCountAnalysis(classifier,data).total

        num_params = sum(p.numel() for p in classifier.parameters()) 
      
        info = {}
        info['train_epoch_time(bsz = 100sample)'] = epoch_time
        info['infer_one_sample_time'] = infer_one_sample_time
        info['flops(infer_1_sample)'] = flops
        info['num_params(infer)'] = num_params
        info['allocated_memory'] = allocated_memory
        info['cached_memory'] = cached_memory
        
        info = pd.DataFrame(info.items())
        info.to_csv(os.path.join(save_path,'info.csv'))



def test(model_name,classifier,data_loader,snrs = None,snr_indexs = None,save_path = None,final = False,model=None,converter = None):
    prediction = []
    true = []
    if model_name == 'HVSF' or model_name == 'HVSF_wo_finetunning':
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
    
    else:
        if final:
            classifier.load_state_dict(torch.load(os.path.join(save_path,'classifier_state.pth')))
        for batch_idx, (data,_, labels) in enumerate(data_loader):
            data = data.cuda()
            labels = labels.cuda()
            result = torch.argmax(classifier(data),dim = 1)
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
               


for rate in [0.01,0.05,0.1,0.3,0.5,0.6]:
#for rate in [0.6]:
    for Time in range(3):
        root_path = os.path.join('/root/autodl-tmp/所有模型比较RML2016a',str(rate),str(Time))
        if not os.path.exists(root_path):
            os.mkdir(root_path)

        bsz = 100
        traindataset,valset,testset,classes,snrs,snr_indexs = get_signal('/root/autodl-tmp/RML2016.10a_dict.pkl','/root/autodl-tmp/CWTdata.pkl',rate,L=1,snrs_index=0)
        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=bsz, shuffle=True, drop_last = True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=bsz, shuffle=False, drop_last = True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=bsz, shuffle=False, drop_last = True)

        #for model_name in ['CNN','MCLDNN','PET','CLDNN','HVSF_wo_finetunning','HVSF']:
        for model_name in ['HVSF']:
            save_path = os.path.join(root_path,model_name)
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            if model_name == 'HVSF' or model_name == 'HVSF_wo_finetunning':
                model = Model(160,16,2).cuda()
                classifier = classifier_head(3*160,11).cuda()
                model.load_state_dict(torch.load(os.path.join('/root/autodl-tmp/所有模型比较RML2016a','model_state.pth')))
                model.train()
                classifier.train()
                converter = video_Converter(bsz,14)
                if model_name == 'HVSF':
                    train(model_name,classifier,50,train_loader,val_loader,save_path,model=model,finetunning = True,converter  = converter)
                else:
                    train(model_name,classifier,50,train_loader,val_loader,save_path,model=model,finetunning = False,converter  = converter)
                test(model_name, classifier, test_loader,snrs, snr_indexs, save_path, final = True, model=model, converter = converter)
            else:
                classifier = get_class(model_name,(2,128),11).cuda()
                train(model_name,classifier,50,train_loader,val_loader,save_path)
                test(model_name, classifier, test_loader,snrs, snr_indexs, save_path, final = True)
