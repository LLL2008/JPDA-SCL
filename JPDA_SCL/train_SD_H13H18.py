import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time
import gc
from tqdm import tqdm
import random
import os

from ImageDataset_SD_H13H18_com_cls import ImgDataset_train
from networks.MDGTnet import MDGTnet
from utils.cls_weight_calculation import weight_calc_HSI
from utils.lr_adjust import lr_adj
from utils.augmented import radiation_noise, flip_augmentation
from sklearn.metrics import accuracy_score
from loss import SupConLoss, Rloss

seeds = [1864]
CLASS_NUM = 4
acc = np.zeros([len(seeds), 1])  
A = np.zeros([len(seeds), CLASS_NUM]) 
k = np.zeros([len(seeds), 1])

for se in range(len(seeds)):
    print('#######################se######################## ', se)
    seed = seeds[se]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    # set model paras
    in_ch = 144
    out_ch = [500, 750, 500, 500, 500, 500, 300, 150]
    spec_range = [65, 144]
    padding = 0
    class_num = 4
    slice_size = 3
    batch_size = 512
    num_epoch = 1
    learning_rate = 0.006
    device = "cuda:0"
    torch.cuda.set_device(0)
    w = [2, 2, 1]

    # load train data H13 and H18
    img_1 = np.load("data/SD_H1318/gen_H13/img_norm_all.npy")
    img_1 = torch.from_numpy(img_1).float()

    img_2 = np.load("data/SD_H1318/gen_H18/img_norm_all.npy")
    img_2 = torch.from_numpy(img_2).float()
    label_1 = np.load("data/SD_H1318/gen_H13/gt_norm_all.npy")
    label_2 = np.load("data/SD_H1318/gen_H18/gt_norm_all.npy")

    label_1 = torch.LongTensor(label_1)
    label_2 = torch.LongTensor(label_2)

    print("img_1", img_1.shape)
    print("img_2", img_2.shape)
    print("label_1", label_1.shape)
    print("label_2", label_2.shape)

    data_and_label = (img_1, label_1, img_2, label_2)
    del img_1, img_2, label_1, label_2

    train_set = ImgDataset_train(*data_and_label)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    del data_and_label
    gc.collect()

    # calculate the number of samples
    label_all = []
    for i, data in enumerate(train_loader):
        for j in range(data[2].shape[0]):
            label_all.append(data[2][j].tolist())
            label_all.append(data[5][j].tolist())

    label_all = torch.tensor(label_all).argmax(dim=1) + 1
    label_all = label_all.numpy()
    weight_cls = weight_calc_HSI(
        label_all, cls_id=list(range(1, class_num + 1)))
    print(weight_cls)

    # define and load model
    model = MDGTnet(in_ch=in_ch, out_ch=out_ch, padding=padding, slice_size=slice_size, spec_range=spec_range,
                    class_num=class_num).to(device)

    # define loss functions
    loss_classify = nn.BCEWithLogitsLoss(
        reduction='mean', pos_weight=weight_cls.to(device))
    ContrastiveLoss_s = SupConLoss(temperature=0.1)
    # train
    time_start = time.time()

    for epoch in range(num_epoch):

        epoch_start_time = time.time()
        train_loss = 0.0

        model.train()

        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        loop_len = len(loop)
        for i, data in loop:
            num_iters = epoch * loop_len + i + 1
            optimizer, learning_rate = lr_adj(num_iters, learning_rate, model)
         
            optimizer.zero_grad()
            d1 = data[1].to(device)
            d0 = data[0].to(device)
            d4 = data[4].to(device)
            d3 = data[3].to(device)
          
            y_out_1, side_1_1, side_1_2, side_1_3, side_1_4 = model(
                d1, d0)
            y_out_2, side_2_1, side_2_2, side_2_3, side_2_4 = model(
                d4, d3)
            del d1, d0, d4, d3
            gc.collect()
            torch.cuda.empty_cache()

            #loss_RCS
            R1 = Rloss(side_1_1, side_2_1, data[2], data[5])
            R2 = Rloss(side_1_2, side_2_2, data[2], data[5])
            R3 = Rloss(side_1_3, side_2_3, data[2], data[5])
            R4 = Rloss(side_1_4, side_2_4, data[2], data[5])

            R = R1+R2+R3+R4
            del R1, R2, R3, R4
            gc.collect()
            torch.cuda.empty_cache()

            #loss_BCE
            bce_weight_1 = torch.full(data[2].size(), 1, device=device)
            bce_weight_1[data[2] > 0.5] = 2
            bce_weight_2 = torch.full(data[5].size(), 1, device=device)
            bce_weight_2[data[5] > 0.5] = 2

            loss1_1 = loss_classify(y_out_1, data[2].float().to(device))
            loss1_2 = loss_classify(y_out_2, data[5].float().to(device))

            loss1 = 1 * torch.mean(bce_weight_1 * loss1_1) + \
                1 * torch.mean(bce_weight_2 * loss1_2)

            del side_1_1, side_1_2, side_1_3, side_1_4, side_2_1, side_2_2, side_2_3, side_2_4
            gc.collect()
            torch.cuda.empty_cache()


             #loss_SCL
            dn1 = radiation_noise(data[1]).type(torch.FloatTensor).to(device)
            dn0 = radiation_noise(data[0]).type(torch.FloatTensor).to(device)

            _, _, _, _, side_1_4c = model(
                dn1, dn0)

            del dn1, dn0
            gc.collect()
            torch.cuda.empty_cache()

            da1 = flip_augmentation(data[1]).to(device)
            da0 = flip_augmentation(data[0]).to(device)

            _, _, _, _, side_1_4a = model(
                da1, da0)
            del da1, da0
            gc.collect()
            torch.cuda.empty_cache()

          
            dn4 = radiation_noise(data[4]).type(torch.FloatTensor).to(device)
            dn3 = radiation_noise(data[3]).type(torch.FloatTensor).to(device)
            _, _, _, _, side_2_4c = model(
                dn4, dn3)
            del dn4, dn3
            gc.collect()
            torch.cuda.empty_cache()
            da4 = flip_augmentation(data[4]).type(torch.FloatTensor).to(device)
            da3 = flip_augmentation(data[3]).type(torch.FloatTensor).to(device)
            _, _, _, _, side_2_4a = model(
                da4, da3)
            del da4, da3
            gc.collect()
            torch.cuda.empty_cache()

            y1 = torch.argmax(data[2], dim=1)
            y1 = y1.squeeze().cuda()
            y2 = torch.argmax(data[5], dim=1)
            y2 = y2.squeeze().cuda()

            all_source_con_features4_1 = torch.cat(
                [side_1_4c.unsqueeze(1), side_1_4a.unsqueeze(1)], dim=1)
            del side_1_4c, side_1_4a
            gc.collect()

            all_source_con_features4_2 = torch.cat(
                [side_2_4c.unsqueeze(1), side_2_4a.unsqueeze(1)], dim=1)
            del side_2_4c, side_2_4a
            gc.collect()

            contrastive_loss_s4_1 = ContrastiveLoss_s(
                all_source_con_features4_1, y1)
            contrastive_loss_s4_2 = ContrastiveLoss_s(
                all_source_con_features4_2, y2)

            contrastive_loss = contrastive_loss_s4_1+contrastive_loss_s4_2

            batch_loss = w[0] * loss1 + w[1] * R + w[2]*contrastive_loss

            batch_loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss += batch_loss.item()
                gt_1 = data[2].argmax(dim=1).flatten().cpu().numpy()
                pred_1 = y_out_1.argmax(dim=1).flatten().cpu().numpy()
                gt_2 = data[5].argmax(dim=1).flatten().cpu().numpy()
                pred_2 = y_out_2.argmax(dim=1).flatten().cpu().numpy()
                oa_1 = accuracy_score(gt_1, pred_1)
                oa_2 = accuracy_score(gt_2, pred_2)

            loop.set_description(f'Epoch [{epoch + 1}/{num_epoch}]')
            loop.set_postfix(classify_loss=loss1.item(), R_loss=R.item(), con_loss=contrastive_loss.item(),
                             batch_loss=batch_loss.item(), lr=optimizer.state_dict()['param_groups'][0]['lr'],
                             oa_1=oa_1, oa_2=oa_2)
            optimizer.zero_grad()
        if epoch == num_epoch-1:
            os.makedirs("./models/Ours_H", exist_ok=True)
            torch.save(model.state_dict(),
                       './models/Ours_H/model{}.pth'.format(se))

        print('[%03d/%03d] %2.2f sec(s) Train  Loss: %3.6f' %
              (epoch + 1, num_epoch, time.time() - epoch_start_time, train_loss))

    time_end = time.time()
    print("training time:", time_end - time_start, 's')
