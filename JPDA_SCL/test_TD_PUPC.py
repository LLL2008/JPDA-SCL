import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time
import gc
from tqdm import tqdm
import random
import os

from networks.MDGTnet import MDGTnet
from ImageDataset_SD_H13H18_com_cls import ImgDataset_test_bce
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from utils.label_vision import label_vision_1d

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

    data_list = ["PC", "PU"]
    data_name = data_list[0]
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

    model_path = r"./models/Ours_H/model{}.pth".format(se)

    # load test data PU or PC
    if data_name == 'PC':
        img = np.load("data/SD_H1318/gen_PC/img_norm_all.npy")
        label = np.load("data/SD_H1318/gen_PC/gt_norm_all.npy")
    else:
        img = np.load("data/SD_H1318/gen_PU/img_norm_all.npy")
        label = np.load("data/SD_H1318/gen_PU/gt_norm_all.npy")

    img = torch.from_numpy(img).float()
    label = torch.LongTensor(label)

    test_set = ImgDataset_test_bce(img, label)
    del img, label
    gc.collect()

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # define and load model
    model = MDGTnet(in_ch=in_ch, out_ch=out_ch, padding=padding, slice_size=slice_size, spec_range=spec_range,
                    class_num=class_num).to(device)
    model.load_state_dict(torch.load(model_path))

    # test
    test_start = time.time()
    correct_num = 0
    gt_total = []
    pred_total = []
    row_col_total = []
    with torch.no_grad():
        loop = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, data in loop:
            y_out_te, __, __, __, __ = model(
                data[1].to(device), data[0].to(device))
            gt_te = data[2][:, :4].argmax(dim=1).flatten().cpu().numpy()
            row_col = data[2][:, 4:]   #
            pred_prob = torch.sigmoid(y_out_te)
            pred = pred_prob.argmax(dim=1).flatten().cpu().numpy()

            gt_total.extend(gt_te)
            pred_total.extend(pred)
            row_col_total.extend(row_col)
            oa_batch = np.sum(gt_te - pred == 0) / data[0].shape[0]

            loop.set_description(f'[{i}/{len(test_loader)}]')
            loop.set_postfix(oa_batch=oa_batch)

    acc[se] = accuracy_score(gt_total, pred_total)
    OA = acc[se]
    C = confusion_matrix(gt_total, pred_total)
    A[se, :] = np.diag(C) / np.sum(C, 1, dtype=np.float32)
    k[se] = cohen_kappa_score(gt_total, pred_total)

    print('\t\tAccuracy: ({:.2f}%)\n', OA)
    os.makedirs("./logs/Ours", exist_ok=True)
    if data_name == "PC":
        label_vision_1d(pred_total, row_col_total, 1096, 715,
                        "./logs/Ours/pred_H1318_PC{}.png".format(se))
        label_vision_1d(gt_total, row_col_total, 1096, 715,
                        "./logs/Ours/gt_H1318_PC{}.png".format(se))
    elif data_name == "PU":
        label_vision_1d(pred_total, row_col_total, 610, 340,
                        "./logs/Ours/pred_H1318_PU{}.png".format(se))
        label_vision_1d(gt_total, row_col_total, 610, 340,
                        "./logs/Ours/gt_H1318_PU{}.png".format(se))

    test_end = time.time()


AA = np.mean(A, 1)  
AAMean = np.mean(AA, 0)
AAStd = np.std(AA)
AMean = np.mean(A, 0)
AStd = np.std(A, 0)
OAMean = np.mean(acc)
OAStd = np.std(acc)
kMean = np.mean(k)
kStd = np.std(k)
os.makedirs("./logs/Ours/results", exist_ok=True)

# Open file to write results
with open(f"./logs/Ours/results/{data_name}_results.txt", "w") as f:
    f.write(f"Test time per se(s): {test_end-test_start:.5f}\n")
    f.write(f"Average OA: {100*OAMean:.2f} +- {100*OAStd:.2f}\n")
    f.write(f"Average AA: {100*AAMean:.2f} +- {100*AAStd:.2f}\n")
    f.write(f"Average kappa: {100*kMean:.4f} +- {100*kStd:.4f}\n")
    f.write("\nAccuracy for each class:\n")
    for i in range(CLASS_NUM):
        f.write(f"Class {i}: {100*AMean[i]:.2f} +- {100*AStd[i]:.2f}\n")

    f.write("\nAccuracy for each seed:\n")
    best_iDataset = 0
    for i in range(len(acc)):
        f.write(f"Seed {i}: {acc[i][0]:.4f}\n")
        if acc[i] > acc[best_iDataset]:
            best_iDataset = i
    f.write(f"\nBest accuracy: {acc[best_iDataset][0]:.4f}")

# Keep console output as well
print(f"Test time per se(s): {test_end-test_start:.5f}")
print(f"Average OA: {100*OAMean:.2f} +- {100*OAStd:.2f}")
print(f"Average AA: {100*AAMean:.2f} +- {100*AAStd:.2f}")
print(f"Average kappa: {100*kMean:.4f} +- {100*kStd:.4f}")
print("Accuracy for each class:")
for i in range(CLASS_NUM):
    print(f"Class {i}: {100*AMean[i]:.2f} +- {100*AStd[i]:.2f}")

best_iDataset = 0
for i in range(len(acc)):
    print(f"{i}:{100*acc[i][0]:.4f}")
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print(f"Best acc all={100*acc[best_iDataset][0]:.4f}")
