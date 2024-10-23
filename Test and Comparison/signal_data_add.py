import numpy as np
import h5py
import torch
import pickle as pk
import torch

def get_signal(dir1,dir2,rate,L=30,snrs_index=0):
    #dir1:原信号
    #dir2,dir3:模态间信号，dir2为小波变换能量图，dir3为星座图
    #dir4:序列模态对比信号，是原信号经time warping变换后的信号

    f1 = open(dir1, 'rb')
    f2 = open(dir2, 'rb')
  
    data = pk.load(f1, encoding='latin1')
    all_snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [1, 0])

    snrs = all_snrs[snrs_index:]
    not_used_snrs = all_snrs[:snrs_index]
    print("使用的snrs: ",snrs)
    print("未使用的snr:",not_used_snrs)
    for mod in mods:
        for snr in not_used_snrs:
            del data[(mod, snr)]
    print("原始信号读取完成")
    CWTdata = pk.load(f2, encoding='latin1')
    for mod in mods:
        for snr in not_used_snrs:
            del CWTdata[(mod, snr)]
    print("CWT读取完成")



    snr_choise = [10]
    X = []
    CWT = []
    lbl = []
    train_idx = []
    lbl_idx = []
    val_idx = []
    data_size = 0
    test_idx=[]

    for mod in mods:
        for snr in snrs:
            length = data[(mod, snr)].shape[0]
            X.append(data.pop((mod, snr)))
            CWT.append(CWTdata.pop((mod, snr)))
            for i in range(length):  lbl.append((mod, snr))
            train_choise = np.random.choice(range(data_size, data_size + length), size=int(length * 0.6 * rate), replace=False)
            train_idx += list(train_choise)
            if snr in snr_choise:
                lbl_idx += list(np.random.choice(train_choise, size=L, replace=False))
           
            val_idx += list(
                np.random.choice(list(set(range(data_size, data_size + length)) - set(train_idx)),
                                 size=int(length * 0.2), replace=False))
            
            test_idx += list(
                np.random.choice(list(set(range(data_size, data_size + length)) - set(train_idx) - set(val_idx)),
                                 size=int(length * 0.2), replace=False))
             
            data_size += length

    
    print("每一类中有{}个训练样本".format(length * 0.6 * rate))

    X = np.vstack(X)
    CWT = np.vstack(CWT)

    #X = np.expand_dims(X, axis=1)
    print("X.shape",X.shape)
    print("CWT.shape",CWT.shape)

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    del X

    CWT_train = CWT[train_idx]
    CWT_val = CWT[val_idx]
    CWT_test = CWT[test_idx]
    del CWT

    Y_train = np.array(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_val = np.array(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
    Y_test = np.array(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))



    traindataset = arr_to_dataset(X_train, CWT_train, Y_train)

    valdataset = arr_to_dataset(X_val, CWT_val, Y_val)

    testdataset = arr_to_dataset(X_test, CWT_test, Y_test)


    snr_index = np.array(list(map(lambda x: lbl[x][1], test_idx)))

    snr_indexs = []
    for snr in snrs:
        snr_indexs.extend(np.where(snr_index == snr))

    return traindataset,valdataset,testdataset, mods,snrs,snr_indexs


def arr_to_dataset(data1, data2, label):
    data1 = torch.from_numpy(data1)
    data2 = torch.from_numpy(data2) 

    
    label = torch.from_numpy(label)
    dataset = torch.utils.data.TensorDataset(data1,data2,label)
    return dataset

