import math
import random
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def get_exp(len=4000):
    singals = []
    labels = []
    for i in range(len):
        k = 0
        l = random.randint(0,7)
        if l == 0:
            phi = math.pi/12
            labels.append(0)
        elif l == 1:
            phi = math.pi/10
            labels.append(1)

        elif l == 2:
            phi = math.pi/8
            labels.append(2)

        elif l ==3:
            phi = math.pi/6
            labels.append(3)
    
        elif l ==4:
            phi = math.pi/4
            labels.append(4)


        elif l ==5:
            phi = math.pi/14
            labels.append(5)

        elif l ==6:
            phi = math.pi/16
            labels.append(6)
            


        else:
            phi = math.pi/2
            labels.append(7)
            
        singal_x = []
        singal_y = []
        for j in range(128):
            k = k + random.randint(0,5)
            singal_x.append(math.cos(k*phi))
            singal_y.append(math.sin(k*phi))
        singals.append([singal_x,singal_y])
    

    Singal = torch.tensor([])
    for i in range(len):
        if i == 0:
            Singal = torch.tensor([singals[i]])
        else:
            Singal = torch.cat([Singal,torch.tensor([singals[i]])],dim=0)
    Labels = torch.tensor(labels)

    return Singal,Labels



def compute_loss(pre,target):
    l2 = torch.mm(torch.norm(target,dim=1).unsqueeze(1),torch.norm(pre,dim=1).unsqueeze(0))
    #pre = pre + 1e-5
    bsz = target.shape[0]
    feature_dim = target.shape[1]
    target = target.unsqueeze(1).expand(bsz, bsz, feature_dim)
    pre = pre.unsqueeze(0).expand(bsz, bsz, feature_dim)
    # 对 A 中每个向量与 B 中每个向量进行点积
    dot_product = torch.matmul(target, pre.transpose(1, 2))
    # 将点积结果保存为矩阵形式
    result = dot_product.squeeze()
    result = result[:,0,:]
    result = torch.div(result,l2)
    result = torch.div(result,0.07)
    #print("result",result)
    result = torch.exp(result)
    #print(result)
    diag = torch.diag(result)
    #print(diag)
    total_lic = torch.sum(result,dim=0)
    #print(total_lic)
    lic = torch.div(diag,total_lic)
    lic = -torch.log(lic)
    #print("lic",lic)
    return torch.sum(lic)/lic.shape[0]


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()


    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




def figure_plot(true_labels,pre_labels,classes,snrs,snr_indexs,figure_path=None):
    acc = {}
    acc_mod_snr = np.zeros((len(classes), len(snrs)))
    for i in range(len(snrs)):
        true_label = true_labels[snr_indexs[i]]
        #print(true_label.shape)
        pre_label = pre_labels[snr_indexs[i]]
        cor = np.sum(true_label == pre_label)
        acc[snrs[i]] = 1.0 * cor / true_label.shape[0]

        plot_confusion_matrix(true_label,pre_label,classes,
                              title="Confusion Matrix (SNR=%d)(ACC=%2f)" % (snrs[i], 100.0 * acc[snrs[i]]),
                              save_filename =os.path.join(figure_path,'Confusion(SNR=%d)(ACC=%2f).png' % (snrs[i], 100.0 * acc[snrs[i]])))
        confnorm_i, _, _ = calculate_confusion_matrix(true_label, pre_label, classes)
        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i) / np.sum(confnorm_i, axis=1), 3)


    plt.plot(snrs, list(map(lambda x: acc[x], snrs)),'.-')


    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("CNN Classification Accuracy on dataset RadioML 2''016.10 Alpha")
    plt.savefig(os.path.join(figure_path,'dB to Noise Ratio'))
    plt.close()

    # plot acc of each mod in one picture
    dis_num = len(classes)
    for g in range(int(np.ceil(acc_mod_snr.shape[0] / dis_num))):
        assert (0 <= dis_num <= acc_mod_snr.shape[0])
        beg_index = g * dis_num
        end_index = np.min([(g + 1) * dis_num, acc_mod_snr.shape[0]])

        plt.figure(figsize=(12, 10))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("Classification Accuracy for Each Mod")

        for i in range(beg_index, end_index):
            plt.plot(snrs, acc_mod_snr[i],'.-', label=classes[i])
            # 设置数字标签
            for x, y in zip(snrs, acc_mod_snr[i]):
                plt.text(x, y, y, ha='center', va='bottom', fontsize=8)

        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(figure_path,'acc_with_mod.png'))
        plt.close()
    return acc,acc_mod_snr


def calculate_confusion_matrix(Y,Y_hat,classes):
    n_classes = len(classes)
    conf = np.zeros([n_classes,n_classes])
    confnorm = np.zeros([n_classes,n_classes])

    for k in range(0,Y.shape[0]):
        i = Y[k]
        j = Y_hat[k]
        conf[i,j] = conf[i,j] + 1

    for i in range(0,n_classes):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    # print(confnorm)

    right = np.sum(np.diag(conf))
    wrong = np.sum(conf) - right
    return confnorm,right,wrong




def plot_confusion_matrix(y_true, y_pred, labels, save_filename=None, title='Confusion matrix'):

    cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0
    for x_test, y_test in zip(x.flatten(), y.flatten()):

        if (intFlag):
            c = cm[y_test][x_test]
            plt.text(x_test, y_test, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')

        else:
            c = cm_normalized[y_test][x_test]
            if (c > 0.01):
                #这里是绘制数字，可以对数字大小和颜色进行修改
                plt.text(x_test, y_test, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')
            else:
                plt.text(x_test, y_test, "%d" % (0,), color='red', fontsize=10, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('Index of True Classes')
    plt.xlabel('Index of Predict Classes')
    plt.savefig(save_filename)
    plt.close()


class video_Converter:
    def __init__(self,batch_bsz,frame_legth):
        self.batch_bsz = batch_bsz
        self.frame_legth = frame_legth
        self.sample_idx = torch.arange(0, batch_bsz).repeat(128, 1).t().reshape(-1).cuda()
        self.frame_idx = torch.arange(0, 128).repeat(1, batch_bsz).squeeze().cuda()
        self.Fundation = torch.zeros(batch_bsz, 128, self.frame_legth, self.frame_legth).cuda()
        self.converter = singal_to_video()
    def __call__(self, singal):
        x = self.converter(singal, self.batch_bsz, self.frame_legth, self.sample_idx, self.frame_idx, self.Fundation)
        x = (x + torch.roll(x, shifts=-1, dims=1) * 0.5 + torch.roll(x, shifts=1, dims=1) * 0.5)
        x = x.unsqueeze(2)
        return x

class singal_to_video(object):
    def __init__(self):
        pass

    def __call__(self, singal, bsz, frame_legth, sample_idx, frame_idx, Fundation):
        lists_for_image = torch.transpose(singal, 1, 2)
        lists_for_image = torch.stack(
            [(a-torch.min(a).item() )/(torch.max(a).item()-torch.min(a).item()) for a in lists_for_image])  # 这里该成了torch.cat 范围是：[-0.5,0.5]
        lists_for_image = torch.round(torch.mul(lists_for_image, frame_legth-1)).to(torch.int) 
        lists_for_image = lists_for_image.reshape(128 * bsz, 2)
        lists_for_image = lists_for_image.long()
        result = Fundation.zero_()
        result[sample_idx, frame_idx, lists_for_image[:, 0], lists_for_image[:, 1]] += 255

        return result