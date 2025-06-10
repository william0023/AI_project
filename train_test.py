#####Resnet34_UNet_SASK_CCD ####
#########跳跃连接###########
#利用ASPP空洞空间卷积池化，添加多尺度特征#
#再利用SK模块，添加卷积选择#
#最后将选择结果与1*1conv/image pool  级联融合#
#再最后的512*8*8跳跃连接中添加SASK模块#
#在Resnet中 用3*3卷积代替maxpool层#
############################
#########Decoder############
#在与encoder来的特征图级联之后 加入ccd选择通道方向模块#
#对级联操作中来自 encoder的浅层纹理信息 与 decoder中的深层语义信息 的很多通道进行选择#
#空洞卷积中添加BN RulE#
#CCD后是3*3卷积降维#

import cv2
import time
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import metrics       #导入评价指标
import torch
from torch import nn

# from tensorboardX import SummaryWriter

from torch.nn import functional as F
from torch import optim           #导入用来更新参数的优化算法
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score
#from RUNet.UNet import Unet
from UNet import *
# from UNet import *
# from SK_RUNet_FFM import *
# from Convext import *
# from unet2022 import *
# from UNet_prama import *
# from UNet_prama512 import *
# from UNet_OCR import *
# from Ghost_UNet_OCR import *
# from UNet_CAM import *
# from CPF_Ghost_GPG_AGNet import  *
# from augmentations import *
#ResNet作UNet的编码器，


image_size_w = 512
image_size_h = 512
data = 'DRIVE' ##数据集名称
# data_dir = r"D:\skx\aa\DRIVE_aug" ##数据集路径
# data_dir = r"/root/autodl-tmp/dataset/CHASEDB1" ##数据集路径
# data_dir = r"/root/autodl-tmp/FIVES" ##数据集路径
data_dir = r"E:/code/python/skx-unet/dataset/DRIVE/DRIVE" ##数据集路径

train_path = os.path.join(data_dir, 'training')
val_path = os.path.join(data_dir, 'test')
model_name = 'UNet' ##网络名称
# dataset = r"D:\skx\aa\DRIVE_aug\{:s}".format(model_name) ##实验数据存放路径
# dataset = r"/root/autodl-tmp/FIVES/results/{:s}".format(model_name) ##实验数据存放路径
dataset = r"E:/code/python/skx-unet/dataset/DRIVE/DRIVE/result/{:s}".format(model_name) ##实验数据存放路径
# dataset = r"/root/autodl-tmp/dataset/DRIVE_aug/result/{:s}".format(model_name) ##实验数据存放路径
save_path = os.path.join(dataset, 'checkpoint')

resume = False  #训练中断是否保存已训练权重，复训时在此之后继续
normalization = False
loss_balance_mode = True



# Net =Unet()
# Net =SK_ASPPUNet34(pretrained=True)
# Net =convnext_base()
# Net = unet2022()
#Net = se_resnet()
#Net = cbam_resnet()
# Net = Unet_OCR()
# Net = Ghost_Unet()
# Net = Unet_CAM()
Net = Unet()

# model = Unet()
# # model = UNext_Bi(num_classes=2)
# model_path = "/root/skx/main/RUNet/Unet1_checkpoint_best_DI.pth.tar"
# print('Loading weights into state dict...')
# model_dict = model.state_dict() # 新模型权重载入
# pretrained_dict = torch.load(model_path)  # 原模型权重载入
# state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
# #print(state_dict)
# #
# model_dict.update(state_dict)
# model.load_state_dict(model_dict)
# print('Finished!')
# #
# Net = model.train()
#


#---超参数---
LR = 0.04
LR_adjust_time = 5
LR_deacy_ratio = 0.1
MOMENTUM = 0.9  # 0.9
WEIGHT_DECY = 5e-4
Batch_Size = 2 ####原设置为6
Epoch = 2    #30
#  ------
training_print_time = 10 ##每10次迭代输出loss
image_resize_w = image_size_w
image_resize_h = image_size_h
num_classes = 2 ###输出类别
Threshold = 1
OPTIMIZER = 'SGD'####SGD&Adam
device = torch.device('cpu') ##cpu or cuda
torch.cuda.empty_cache()
path = os.getcwd()

augmentation_mode = False

train_losses = []

###——————随机数据增强——————
# data_augmentations = Compose([
#     RandomRotate(10),
#     RandomHorizontallyFlip(),
#     RandomVerticallyFlip(),
# ])

# class data_augmentations(Dataset):
#     res=[]
#     img_np = np.array(Dataset)  # 转化为矩阵
#     width, height = img_np.shape  # 获取宽与高
#     N = np.zeros(shape=(1, 256), dtype='float')  # 构造零矩阵，用以统计像素数
#
#     # 遍历各个灰度值统计个数
#     for i in range(0, width):
#         for j in range(0, height):
#             k = img_np[i, j]
#             N[0][k] = N[0][k] + 1
#
#     N = N.flatten()  # 扁平化
#
#     # 线性变化
#     J = img_np.astype('float')
#     J = 0 + (J - 42) * (255 - 0) / (232 - 42)  # 利用公式转换
#     # 像素值小于0或大于255的分别赋值为0和255
#     for i in range(0, width):
#         for j in range(0, height):
#             if J[i, j] < 0:
#                 J[i, j] = 0
#             elif J[i, j] > 255:
#                 J[i, j] = 255
#     image_1 = J.astype('uint8')
#     J = J.flatten()
#     res.append(image_1)
#


class Load_dataset(Dataset):  # 加载数据集
    def __init__(self, input_path, target_path, to_tensor=True,
                 img_size_w=image_resize_w, img_size_h=image_resize_h, augmentations=None, normalization=False):
        self.input_path = input_path
        self.target_path = target_path
        self.to_tensor = to_tensor
        self.augmentations = augmentations
        self.normalization = normalization
        self.img_size_w = img_size_w
        self.img_size_h = img_size_h
        self.img_list = os.listdir(self.input_path)  # s.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
        self.lab_list = os.listdir(self.target_path)

    def __getitem__(self, index):       #_getitem__函数的作用是根据索引index遍历数据，__getitem__必须根据index返回响应的值，该值会通过index传到dataloader中进行后续的batch批处理。
        # image = Image.open(os.path.join(
        #     self.input_path, self.img_list[index]))
        #print(image.shape)
        image = Image.open(os.path.join(
                self.input_path, self.img_list[index])).convert('RGB')      #将原图像打开转化为RGB格式
        label = Image.open(os.path.join(                                    #将目标图像打开转化为灰度格式
                self.target_path, self.lab_list[index])).convert('L')
        imgarr = np.array(image)              # 输出的效果：imgarr.shape=[width,height,channels]
        #print(imgarr.shape)
        labarr = np.array(label)
        if self.augmentations is not None:
            imgarr, labarr = self.augmentations(imgarr, labarr)
        if self.to_tensor:
            image, label = self.totensor(imgarr, labarr)
        if normalization:
            image = transforms.Normalize(
                # mean=[0.485,0.456,0.406],
                #  std=[0.229,0.224,0.225] ###ImageNet 归一化
                mean=[0.5,0.5,0.5],
                 std=[0.5,0.5,0.5])(image)  # 对图像数据进行标准化处理，即将图像的每个通道减去均值并除以标准差，使得数据分布更加接近标准正态分布。
        return image, label

    def crop(image, dx):
        list = []
        for i in range(image.shape[0]):
            for x in range(image.shape[1] // dx):
                for y in range(image.shape[2] // dx):
                    list.append(image[i, y * dx: (y + 1) * dx,
                                x * dx: (x + 1) * dx])  # 这里的list一共append了20x12x12=2880次所以返回的shape是(2880,48,48)
        return np.array(list)

    def totensor(self, img, lab):
        img = Image.fromarray(img)   #实现array到image
        img = img.resize((self.img_size_w, self.img_size_h))   #变换image图片大小
        img = np.array(img)  #换成numpy数组
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img)   #变成tensor张量
        lab = Image.fromarray(lab)
        lab = lab.resize((self.img_size_w, self.img_size_h))
        lab = np.array(lab)
        # lab[lab==175] = -1
        lab[lab<110] = 0 ###background
        lab[lab>=110] = 1 ###mass
        lab = lab.astype(int)
        #imshow(img)
        lab = (torch.from_numpy(lab)).long()
        return img, lab
    def __len__(self):     #__len__函数的作用是返回数据集的长度
        return len(self.img_list)


class RunningScore(): ###指标计算
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    # 计算混淆矩阵 hist
    def _fast_hist(self, label_true, label_pred, num_class):
        mask = (label_true >= 0) & (label_true < num_class)
        hist = np.bincount(
                num_class*label_true[mask].astype(int) +
                label_pred[mask], minlength=num_class*2).reshape(num_class, num_class)   ##调用了fast_hist,传入label_trues, label_preds，将混淆矩阵赋值给confusion_matrix
        return hist
    
    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), 
                                                     lp.flatten(),
                                                     self.num_classes)
    ####################################################
    def calcAUC_byProb(self, labels, probs):
        labels = labels.flatten()
        probs = probs.flatten()
        self.auc = roc_auc_score(labels, probs)


    def get_scores(self):
        hist = self.confusion_matrix

        acc = np.diag(hist).sum() / hist.sum()     # PA
        acc_cls = np.diag(hist) / hist.sum(axis=1)#按行相加
        acc_row = np.diag(hist) / hist.sum(axis=0)#按列相加
        acc_mean = np.nanmean(acc_cls)             #MPA
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        DI = 2*np.diag(hist)[1]/(2*np.diag(hist)[1]+hist[0][1]+hist[1][0])      #求的DICE
        mean_iu = np.nanmean(iu)      #求的MIOU
        freq = hist.sum(axis=1) / hist.sum()
        fwiu = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.num_classes), iu))
        sen = np.diag(hist)[1] / hist.sum(axis=0)[1]
        spe = np.diag(hist)[0] / hist.sum(axis=0)[0]
        ppv = np.diag(hist)[1] / hist.sum(axis=1)[1]
        F1 = 2*ppv*sen / (ppv+sen)
        JS = np.diag(hist)[1] / ((2*np.diag(hist)[1]+hist[0][1]+hist[1][0]) - np.diag(hist)[1])
        auc = self.auc
        return {
                '  PA : \t': acc,#准确度，找得对
                ' MPA : \t': acc_mean,
                ' TNR : \t' : acc_row[0],
                ' TPR : \t' : acc_row[1],#召回率recall，找得全
                # ' FPR : \t' : 1-acc_cls[0],
                ' PPV : \t' : acc_cls[1], #Positive predictive value:TP/(TP+FP)
                # ' NPV : \t' : acc_row[1],
                #'FreqW Acc : \t': fwavacc,
                # 'PIoU : \t': cls_iu[1],
                # 'NIoU : \t': cls_iu[0],
                 'MIoU : \t' : mean_iu,
                # 'FWIoU: \t' : fwiu,
                ' Sen : \t': sen,#
                ' Spe : \t' : spe,
                ' ppv : \t' : ppv,
                'F1_scores : \t' : F1,
                'Jaccard_similar : \t' : JS,
                '  DI : \t' : DI,
                'AUC : \t' : auc,
                }

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))


def Segment():
    save_name = '_current_checkpoint.pth.tar'
    start_epoch = 0
    end_epoch = Epoch
    locate_epoch = start_epoch + 1
    best_DI = 0.0

    if augmentation_mode == True:
        data_aug = data_augmentations
    else:
        data_aug = None

    if normalization == True:
        norm = True
    else:
        norm = False

    train_datasets = Load_dataset(input_path=os.path.join(train_path, 'images'),
                                  target_path=os.path.join(train_path,'1st_manual'),
                                  img_size_w=image_resize_w,
                                  img_size_h=image_resize_h,
                                  augmentations=data_aug,
                                  normalization=norm)
    val_datasets   = Load_dataset(input_path=os.path.join(val_path, 'images'),
                                  target_path=os.path.join(val_path,'1st_manual'),
                                  img_size_w=image_resize_w,
                                  img_size_h=image_resize_h,
                                  normalization=norm)
    
    train_dataloader = DataLoader(train_datasets, batch_size=Batch_Size, shuffle=True)
    val_dataloader = DataLoader(val_datasets, batch_size=1, shuffle=False)
    
    # Setup Metrics
    running_metrics = RunningScore(num_classes)

    if not os.path.exists(dataset):
        os.makedirs(dataset)
    if not os.path.exists(os.path.join(save_path,model_name,dataset)):
        os.mkdir(os.path.join(save_path,model_name,dataset))

    net = Net.to(device)

    print(net)
    print('>' * 15, ' Augmentation :{:}'.format(augmentation_mode), '<' * 15)
    print('>' * 15, ' Normalization:{:}'.format(normalization), '<' * 15)
    print('>' * 15, ' Loss Balance :{:}'.format(loss_balance_mode), '<' * 15)
    # print('>' * 15, 'Loss_Threshold:{:}'.format(confusion_threshold), '<' * 15)


    optimizer = optim.SGD(net.parameters(), lr=LR,momentum=MOMENTUM, weight_decay=WEIGHT_DECY)   #优化器

    if OPTIMIZER == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_adjust_time,gamma=LR_deacy_ratio)

    # scheduler = PolynomialLRDecay(optimizer, max_decay_steps=600)

    #resume training checkpoint
    if resume:
        os.chdir(os.path.join(save_path,model_name,dataset))
        if os.path.isfile(model_name+save_name):
            print("==> loading checkpoint '{}'".format(save_name))
            checkpoint = torch.load(model_name+save_name)    #加载模型，存某一次训练采用的优化器、epochs等信息，可将这些信息组合起来构成一个字典，然后将字典保存起来
            start_epoch = checkpoint['epoch']
            locate_epoch = checkpoint['locate_epoch']
            best_DI = checkpoint['best_DI']
            net.load_state_dict(checkpoint['state_dict'],False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("==> Loaded checkpoint '{}' (Epoch {})" .format(save_name, checkpoint['epoch']+1))
        else:
            print("==> No checkpoint found at '{}'".format(model_name+save_name))
        os.chdir(path)

    for epoch in range(start_epoch, end_epoch):        #最后循环该dataloader ，拿到数据放入模型进行训练：
        train(train_dataloader, optimizer, scheduler, net, epoch, end_epoch, running_metrics)
        val_DI,val_images,val_outputs,val_preds,val_labels = val(val_dataloader, net, epoch, end_epoch, running_metrics)
        visualize(epoch, val_images, val_outputs, val_labels)
        is_best = val_DI >= best_DI
        best_DI = max(val_DI, best_DI)
        if is_best:
            best_DI = max(val_DI, best_DI)
            locate_epoch = epoch + 1
        print('#'*15,'Best DI at Epoch {:d}:{:.4f}'.format(locate_epoch,best_DI),15*'#')
        with open(os.path.join(save_path, model_name, dataset, 'val.txt'), 'a') as f:
            print('#'*15, 'Best DI at Epoch {:d}:{:.4f}'.format(locate_epoch, best_DI), 15 * '#', file=f)
        os.chdir(os.path.join(save_path,model_name,dataset))
        save_checkpoint({
            'epoch':epoch+1,
            'locate_epoch':locate_epoch,
            'model_name':model_name,
            'best_DI':best_DI,
            'state_dict':net.state_dict(),
            'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
        }, net, is_best, model_name+save_name, model_name)
        os.chdir(path)


# class SoftDiceLoss(nn.Module):
#     '''
#     Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
#     eps is a small constant to avoid zero division,
#     '''
#
#     def __init__(self, weight=None):
#         super(SoftDiceLoss, self).__init__()
#         self.activation = nn.Softmax2d()
#
#     def forward(self, y_preds, y_truths, eps=1e-8):
#         '''
#         :param y_preds: [bs,num_classes,768,1024]
#         :param y_truths: [bs,num_calsses,768,1024]
#         :param eps:
#         :return:
#         '''
#         bs = y_preds.size(0)
#         num_classes = y_preds.size(1)
#         dices_bs = torch.zeros(bs, num_classes)
#         for idx in range(bs):
#             y_pred = y_preds[idx]  # [num_classes,768,1024]
#             y_truth = y_truths[idx]  # [num_classes,768,1024]
#             intersection = torch.sum(torch.mul(y_pred, y_truth), dim=(1, 2)) + eps / 2
#             union = torch.sum(torch.mul(y_pred, y_pred), dim=(1, 2)) + torch.sum(torch.mul(y_truth, y_truth), dim=(1, 2)) + eps
#
#             dices_sub = 2 * intersection / union
#             dices_bs[idx] = dices_sub
#
#         dices = torch.mean(dices_bs, dim=0)
#         dice = torch.mean(dices)
#         dice_loss = 1 - dice
#         return dice_loss


class SoftDiceLoss(nn.Module):
    #  1 -  / |X| + |Y|
    #将|x n Y | 近似为预测图每个类别score和target之间的点乘，并将结果函数中的元素相加。
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=1) # 增加为2通道

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1
        # probs = torch.sigmoid(logits,targets)
        probs = torch.softmax(logits,1)  #这里的logits表示输入张量，dim表示softmax函数作用的维度。softmax计算概率，之和为1
        probs = probs[:,1,:,:]
        m1 = probs.view(num, -1) #重新构建这个张量的维度，-1代表不在这个维度做限制
        # targets = self.conv(targets) # 增加为2通道
        # m2 = targets.view(num, -1) # win
        m2 = targets.view(num, -1).to(dtype=torch.float32) # lin

        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score



# def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):
#     r""" computational formula：
#         dice = (2 * tp) / (2 * tp + fp + fn)
#     """
#
#     if activation is None or activation == "none":
#         activation_fn = lambda x: x
#     elif activation == "sigmoid":
#         activation_fn = nn.Sigmoid()
#     elif activation == "softmax2d":
#         activation_fn = nn.Softmax2d()
#     else:
#         raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")
#
#     pred = activation_fn(pred)
#
#     N = gt.size(0)
#     pred_flat = pred.view(N, -1)
#     gt_flat = gt.view(N, -1)
#
#     tp = torch.sum(gt_flat * pred_flat, dim=1)
#     fp = torch.sum(pred_flat, dim=1) - tp
#     fn = torch.sum(gt_flat, dim=1) - tp
#     loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
#     return loss.sum() / N
#
#
# class SoftDiceLoss(nn.Module):
#     __name__ = 'dice_loss'
#
#     def __init__(self, activation='sigmoid'):
#         super(SoftDiceLoss, self).__init__()
#         self.activation = activation
#
#     def forward(self, y_pr, y_gt):
#         return 1 - diceCoeffv2(y_pr, y_gt, activation=self.activation)


# class BinaryDiceLoss(nn.Module):
#     def __init__(self):
#         super(BinaryDiceLoss, self).__init__()
#
#     def forward(self, input, targets):
#         # 获取每个批次的大小 N
#         N = targets.size()[0]
#         # 平滑变量
#         smooth = 1
#         # 将宽高 reshape 到同一纬度
#         input_flat = input.view(N, -1).permute(1,0)
#         targets_flat = targets.view(N, -1)
#
#         # 计算交集
#         intersection = input_flat * targets_flat #一维数组中，数值不匹配的问题 无法计算
#         N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
#         # 计算一个批次中平均每张图的损失
#         loss = 1 - N_dice_eff.sum() / N
#         return loss


# class DiceLoss(nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()
#
#     def forward(self, input, target):
#         N = target.size(0)
#         smooth = 1
#
#         input_flat = input.view(N, -1)
#         target_flat = target.view(N, -1)
#
#         intersection = input_flat * target_flat
#
#         loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
#         # loss = 1 - loss.sum() / N
#         return 1 - loss

# class DiceCoeff(nn.Module):
#     """Dice coeff for individual examples"""
#
#     def forward(self, input, target):
#         self.save_for_backward(input, target)
#         self.inter = torch.dot(input.view(-1), target.view(-1)) + 0.0001
#         self.union = torch.sum(input) + torch.sum(target) + 0.0001
#
#         t = 2 * self.inter.float() / self.union.float()
#         return t
#
#     # This function has only a single output, so it gets only one gradient
#     def backward(self, grad_output):
#
#         input, target = self.saved_variables
#         grad_input = grad_target = None
#
#         if self.needs_input_grad[0]:
#             grad_input = grad_output * 2 * (target * self.union + self.inter) \
#                          / self.union * self.union
#         if self.needs_input_grad[1]:
#             grad_target = None
#
#         return grad_input, grad_target
#
#
# def dice_coeff(input, target):
#     """Dice coeff for batches"""
#     if input.is_cuda:
#         s = torch.FloatTensor(1).cuda().zero_()
#     else:
#         s = torch.FloatTensor(1).zero_()
#
#     for i, c in enumerate(zip(input, target)):
#         s = s + DiceCoeff().forward(c[0], c[1])
#
#     return s / (i + 1)


def train(train_dataloader, optimizer, scheduler, net, epoch, end_epoch, running_metrics): ###训练
    #net.train()
    torch.backends.cudnn.enabled = False  #这个设置主要用于实验过程的可复现
    print('>'*15,'TRAINING',time.strftime('%Y-%m-%d %H:%M:%S'),15*'<')
    temp  = os.path.join(save_path, model_name, dataset, 'train.txt')
    with open(os.path.join(save_path, model_name, dataset, 'train.txt'), 'a') as f:
        print('>' * 15, 'TRAINING', time.strftime('%Y-%m-%d %H:%M:%S'), 15 * '<',file=f)
    train_loss = 0
    last_time = time.time()
    # adjust_learning_rate(optimizer, epoch)
    if epoch != 0:
        scheduler.step()
    for param_group in optimizer.param_groups:
        print('='*15,'Current Learning Rate:{:.1e}'.format(param_group['lr']),15*'=')
        with open(os.path.join(save_path, model_name, dataset, 'train.txt'), 'a') as f:
            print('=' * 15, 'Current Learning Rate:{:.1e}'.format(param_group['lr']), 15 * '=', file=f)

    for step, (images, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        if loss_balance_mode == True:
            loss_weight = Loss_Weight(labels).to(device)
        else:
            loss_weight = None
        # ce_loss = nn.CrossEntropyLoss(weight=loss_weight)
        # loss = ce_loss(outputs, labels)
        dice_loss = SoftDiceLoss()
        loss = dice_loss(outputs, labels)
        # loss = SoftDiceLoss(outputs, labels)
        # loss = dice_loss(outputs.max(1)[1], labels)
        loss.requires_grad_(True) #默认为Fulse
        loss.backward() #反向传播求梯度，中间变量全施放以便下次循环
        optimizer.step()
        train_loss += loss.item()
        if (step+1) % training_print_time == 0:
            train_loss = train_loss / training_print_time
            train_time_end = time.time()
            train_time = train_time_end - last_time
            
            # print("Epoch [%d/%d]  Step: %d  Time: %.2fs  Loss: %.4f"
            #       % (epoch+1, end_epoch, step+1, train_time, train_loss))

            print("Epoch:[{:d}/{:d}]||Step:{:d}||Loss:{:.4f}||Time:{:.2f}s".
                  format(epoch+1,end_epoch,step+1,train_loss,train_time))

            #record data
            with open(os.path.join(save_path,model_name,dataset,'train.txt'),'a') as f:
                print("Epoch:[{:d}/{:d}]||Step:{:d}||Loss:{:.4f}||Time:{:.2f}s".
                      format(epoch + 1, end_epoch, step + 1, train_loss, train_time),
                      file=f)
            
            last_time = train_time_end
            train_loss = 0.0
        train_losses.append(train_loss / len(train_dataloader))






def val(val_dataloader, net, epoch, end_epoch, running_metrics): ##测试
    #net.eval()
    print('>' * 15, 'TESTING', time.strftime('%Y-%m-%d %H:%M:%S'), 15 * '<')
    # iterition = 0
    val_images = []
    val_labels = []
    val_outputs = []
    val_preds = []

    #with torch.no_grad():
    for image, label in val_dataloader:
        image = image.to(device)
        label = label.to(device)
        # with torch.no_grad(): ###############不计算梯度##########
        output = net(image)  # 原码
        pred = output.data.max(1)[1].cpu().numpy()
        # pred = pred_Threshold(pred, Threshold)
        # pred_arr = pred.numpy()
        tag = label.data.cpu().numpy()
        running_metrics.update(tag, pred)
        running_metrics.calcAUC_byProb(tag, pred)
        val_images.append(image.data.cpu())
        val_labels.append(label.data.cpu())
        val_outputs.append(output.data.cpu())
        val_preds.append(pred)


    score = running_metrics.get_scores()
    for k, v in score.items():
        print(k, v)
    with open(os.path.join(save_path, model_name, dataset, 'val.txt'), 'a') as f:
        print('Epoch [%d/%d]' % (epoch + 1, end_epoch), file=f)
        for k, v in score.items():
            print(k, v, file=f)
    running_metrics.reset()

    return score['  DI : \t'], val_images, val_outputs, val_preds, val_labels

def count(array, value):
    num = 0
    for i in array:
        for j in i:
            if j == value:
                num += 1
    return num

def Loss_Weight(labels):####loss balance ratio
    labels = labels.data.cpu().numpy()
    black = 0
    white = 0
    for label in labels:
        black += count(label, 0)
        white += count(label, 1)
        #print(black,white)
    weight = [white, black]
    #print(ratio)
    return torch.FloatTensor(weight/np.min(weight))

def pred_Threshold(pred, Threshold):
    pred = pred.squeeze(0)   #使[[1,2,3]] 变成  [1,2,3]  张量必须是1*n 维的，squeeze就是降维的
    pred = pred.astype(int)  #正如astype的中文意思，作为布尔类型，把数组里面的数值全部换成0或者1，0代表False，就是0， 非0代表True，不是0！
    num_original = count(pred, 1)  #数pred里面有多少个1，就是计算pred里面有多少个非0的数值
    num_changed = num_original - int(Threshold * num_original)
    num_change = 0
    # print(num_changed - num_change)
    while num_changed - num_change > 0:
        pred, num_change = del_edge(pred, num_original)
    # print(pred)
    return (torch.from_numpy(pred).unsqueeze(0)).long()   #from_numpy()用来将数组array转换为张量Tensor，unsqueeze(0)在第0维扩展，[10]——[1,10],long() 函数将数字或字符串转换为一个长整型。


def del_edge(pred, num_original):
    b = np.zeros([pred.shape[0], pred.shape[1]], dtype=np.int)
    num_b = 0
    for i in range(1, len(pred)-1):
        for j in range(1, len(pred)-1):   
            if pred[i-1][j]+pred[i+1][j]+pred[i][j-1]+pred[i][j+1] == 4:
                b[i][j] = 1
                num_b += 1
    num_change = num_original - num_b
    # num_original = num_b
    #print(num_change)
    return b, num_change
    

def visualize(epoch, images, outputs, labels):
    
    if not os.path.exists(os.path.join(save_path,model_name,dataset,'epoch_%s'%(epoch+1))):
        os.mkdir(os.path.join(save_path,model_name,dataset,'epoch_%s'%(epoch+1)))
    
    for i in range(len(images)):
        j = i + 1
        images[i] = tensor_to_numpy(images[i])
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        image = Image.fromarray(images[i])
        image.save(os.path.join(save_path,model_name,dataset,'epoch_%s'%(epoch+1))+'/'+
                   'image_%d'%j+'.tif')
        output = Image.fromarray(tensor_to_numpy_output(outputs[i].max(1)[1]))
        output.save(os.path.join(save_path,model_name,dataset,'epoch_%s'%(epoch+1))+'/'+
                    'output_%d'%j+'.tif')
        label = Image.fromarray(tensor_to_numpy(labels[i]))
        label.save(os.path.join(save_path,model_name,dataset,'epoch_%s'%(epoch+1))+'/'+
                   'label_%d'%j+'.tif')
        # pred = Image.fromarray(tensor_to_numpy(preds[i]))
        # pred.save(os.path.join(save_path,model_name,dataset,'epoch_%s'%(epoch+1))+'/'+
        #            'prediction_%d'%j+'.png')
        image_ = cv2.imread(os.path.join(save_path,model_name,dataset,'epoch_%s'%(epoch+1))
                                    +'/'+'image_%d'%j+'.tif')
        label_ = cv2.imread(os.path.join(save_path,model_name,dataset,'epoch_%s'%(epoch+1))
                                    +'/'+'label_%d'%j+'.tif')
        output_ = cv2.imread(os.path.join(save_path,model_name,dataset,'epoch_%s'%(epoch+1))
                                    +'/'+'output_%d'%j+'.tif')
        # pred_ = cv2.imread(os.path.join(save_path,model_name,dataset,'epoch_%s'%(epoch+1))
        #                             +'/'+'prediction_%d'%j+'.png')
        make_outline(epoch, j, image_, label_, output_)



def make_outline(epoch, i, image, label, pred):
    
    label = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)  #to gray
    ret, label = cv2.threshold(label, 110, 255, cv2.THRESH_BINARY)   #to binary
    contours, hierarchy = cv2.findContours(label,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    image = cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    pred = cv2.cvtColor(pred,cv2.COLOR_BGR2GRAY)  #to gray
    ret, pred = cv2.threshold(pred, 110, 255, cv2.THRESH_BINARY)   #to binary
    contours, hierarchy = cv2.findContours(pred,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    image = cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

    # output = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)  #to gray
    # ret, output = cv2.threshold(output, 200, 255, cv2.THRESH_BINARY)   #to binary
    # contours, hierarchy = cv2.findContours(output,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    # image = cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
    
    return(cv2.imwrite(os.path.join(save_path,model_name,dataset,'epoch_%s'%(epoch+1))+'/'+
                   'outline_%d'%i+'.tif', image))



def save_checkpoint(state, model, is_best, filename, model_name):
    torch.save(state, filename)
    if is_best:
        best_model = copy.deepcopy(model.state_dict())
        torch.save(best_model, model_name+'_checkpoint_best_DI.pth.tar')
        # shutil.copyfile(filename, model_name+'_checkpoint_best_DI.pth.tar')


def tensor_to_numpy(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0)
    if img.ndim == 3:
        return img.transpose((1, 2, 0))
    elif img.ndim == 2:
        return img
    
def tensor_to_numpy_output(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0)
    img[img<200] = 0
    img[img>=200] = 255
    #print(img)
    if img.ndim == 3:
        return img.transpose((1, 2, 0))
    elif img.ndim == 2:
        return img

    
def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1.0):
#         self.smooth = smooth
#         super(DiceLoss, self).__init__()
#     def forward(self, input, target):
#         b = input.shape[0]
#         dice_loss = 0.0
#         x = torch.split(input, 1,dim=0)
#         y = torch.split(target,1,dim=0)
#         for i in range(b):
#             iflat = x[i].view(-1)
#             tflat = y[i].view(-1)
#             intersection = (iflat * tflat).sum()
#             loss = 1 - ((2.0 * intersection + self.smooth)/(iflat.sum() + tflat.sum() + self.smooth))
#             dice_loss = dice_loss + loss
#         return torch.mean(dice_loss)

class PolynomialLRDecay(optim.lr_scheduler._LRScheduler):
    #多项式衰减，即得到的学习率为初始学习率和给定最终学习之间由多项式计算权重定比分点的插值。
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=0.9):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.t_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                self.end_learning_rate for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) *
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr


if __name__ == '__main__':
    Segment()
    plt.plot(np.arange(len(train_losses)), train_losses, label="train loss")
    plt.legend()  # 显示图例
    plt.xlabel('epoches')
    # plt.ylabel("epoch")
    plt.title('Model accuracy&loss')
    plt.show()