import argparse
from sklearn import metrics
import torch
import sys
import numpy as np
import os
import time
import logging
import shutil
import copy
from model.resnet import ResNet50_fc, ResNet50_fc2
import random
import pandas as pd
from dataset.alb_dataset import Tumor_dataset_pseudo, Tumor_dataset_val, get_loader, get_loader_resample
from PIL import Image
from transformers import AutoModelForImageClassification
import torch.nn.functional as F


def get_files_redistribution(data_root):
    new_file_pos = []
    new_file_neg = []
    img_names = os.listdir(data_root+'images')
    for img_name in img_names:
        image_root = data_root+'images/'+img_name
        label_root = data_root+'labels/'+img_name
        new_sample = {'img': image_root, 'label': label_root}
        label_sample = np.array(Image.open(label_root))
        # print(np.unique(label_sample))
        if np.max(label_sample) == 1:
            new_file_pos.append(new_sample)
        else:
            new_file_neg.append(new_sample)
    print('pos:', len(new_file_pos), 'neg:',len(new_file_neg))
    return new_file_pos, new_file_neg

def get_files_csv(data_csv):
    data = pd.read_csv(data_csv)
    data_name = data.iloc[:, 0]
    data_pseudo_label = data.iloc[:, 1]
    data_pseudo_label = np.array(data_pseudo_label).astype(np.uint8)
    data_true_label = data.iloc[:, 2]
    data_true_label = np.array(data_true_label).astype(np.uint8)
    data_name = data_name.to_list()
    new_file = [{"img": img, "p_label": p_label, "t_label":t_label} for img, p_label, t_label \
                in zip(data_name, data_pseudo_label, data_true_label)]

    new_file_pos = []
    new_file_neg = []
    for sample in new_file:
        if sample['p_label'] == 1:
            new_file_pos.append(sample)
        else:
            new_file_neg.append(sample)
    # here resample
    return new_file, new_file_neg, new_file_pos

def get_files(data_root):
    new_file = []
    img_names = os.listdir(data_root+'images')
    for img_name in img_names:
        image_root = data_root+'images/'+img_name
        label_root = data_root+'labels/'+img_name
        new_sample = {'img': image_root, 'label': label_root}
        new_file.append(new_sample)
    return new_file

def get_arguments():
    parser = argparse.ArgumentParser(
        description="xxxx Pytorch implementation")
    parser.add_argument("--num_class", type=int, default=2, help="Train class num")
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--crop_size", default=224)
    parser.add_argument("--gpu", nargs="+", type=int)
    parser.add_argument("--batch_size", type=int, default=128, help="Train batch size")
    parser.add_argument("--num_workers", default=6)
    parser.add_argument("--portion", default=1, type=float)
    return parser.parse_args()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch(42)
    args = get_arguments()
    l = logging.getLogger(__name__)
    fileHandler = logging.FileHandler('log/train_HPH.log', mode='a')
    l.setLevel(logging.INFO)
    l.addHandler(fileHandler)

    # load model
    model = ResNet50_fc2().cuda()
 
    # load dataset
    train_all_root = '/home/ubuntu/data/lanfz/datasets/RINGS/train-100-patch/'
    train_data_root = '/home/ubuntu/data/lanfz/codes/adapater_weakly/pseudo_data/HPH_un.csv'
    val_data_root = '/home/ubuntu/data/lanfz/datasets/RINGS/val-patch/'
    test_data_root = '/home/ubuntu/data/lanfz/datasets/RINGS/test-patch/'

    train_files_confidence, train_neg, train_pos = get_files_csv(train_data_root)
    print('pos:',len(train_pos), 'neg:',len(train_neg))
    train_files = train_pos + train_neg
    np.random.shuffle(train_files)
    train_all_files_pos, train_all_files_neg  = get_files_redistribution(train_all_root)
    # print(len(train_all_files_neg), len(train_all_files_pos))
    train_all_files = train_all_files_pos + train_all_files_neg
    val_files = get_files(val_data_root)
    test_files = get_files(test_data_root)
 
    val_files, test_files = val_files+test_files, val_files+test_files
    np.random.shuffle(val_files)

    print(f'train set len:{len(train_files)}')
    l.info(f'train set len:{len(train_files)}')
    l.info(f'val set len:{len(val_files)}')
    l.info(f'test set len:{len(test_files)}')

    train_dataset = Tumor_dataset_pseudo(args, files=train_files)
    train_all_set = Tumor_dataset_val(args, files=train_all_files)
    test_train_dataset = Tumor_dataset_pseudo(args, files=train_files)
    val_dataset = Tumor_dataset_val(args, files=val_files)
    test_dataset = Tumor_dataset_val(args, files=test_files)
    train_loader = get_loader(args, train_dataset)
    # train_loader = get_loader_resample(args, train_dataset, weights=weights)
    train_all_loader = get_loader(args, train_all_set)
    test_train_loader = get_loader(args, test_train_dataset)
    validation_loader = get_loader(args, val_dataset)
    test_loader = get_loader(args, test_dataset)

    epochs = 200

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=8e-4, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs*3//4], gamma=0.1)
    
    l.info(f"Start resnet training for {epochs} epochs")
    max_val_accuracy = 0
    max_epoch = -1
    best_model = None
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(epochs+1):
        train_accuracy = 0
        model.train()
        with torch.cuda.amp.autocast():
            for counter, sample in enumerate(train_loader):
                x_batch = sample['img'].cuda()
                y_batch = sample['p_label'].cuda()
                y_true = sample['t_label'].cuda()
                
                logits = model(x_batch)
                loss = F.cross_entropy(logits, y_batch)

                top1 = accuracy(logits, y_batch, topk=(1,))
                train_accuracy += top1[0]

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            train_accuracy /= (counter + 1)
        val_accuracy = 0
        model.eval()
        with torch.no_grad():
            for counter, sample in enumerate(validation_loader):
                x_batch = sample['img'].cuda()
                y_batch = sample['cls_label'].cuda()

                logits = model(x_batch)
                top1 = accuracy(logits, y_batch, topk=(1,))
                val_accuracy += top1[0]
        val_accuracy /= (counter + 1)

        if epoch % 20 == 0:
            l.info(f"Time:{time.strftime('%H:%M:%S', time.localtime())} epoch:{epoch} Train Accuracy: {train_accuracy.item():.3f} \
                   Val Accuracy: {val_accuracy.item():.3f}")
        
        if val_accuracy.item() > max_val_accuracy:
            max_val_accuracy = val_accuracy.item()
            max_epoch = epoch
            best_model = copy.deepcopy(model)

    test_accuracy = 0
    best_model.eval()

    with torch.no_grad():
        pred_all, gt_all = torch.zeros((1, )), torch.zeros((1, ))
        pred, gt = np.zeros((2,)), np.zeros((2,))
        for counter, sample in enumerate(test_loader):
            x_batch = sample['img'].cuda()
            y_batch = sample['cls_label'].cuda()

            logits = best_model(x_batch)

            top1 = accuracy(logits, y_batch, topk=(1,))
            test_accuracy += top1[0]

            logits_hard = torch.argmax(logits, dim=1)
            gt_all = torch.cat((gt_all, y_batch.cpu()), dim=0)
            pred_all = torch.cat((pred_all, logits_hard.cpu()), dim=0)
            for i in range(logits.shape[0]):
                gt[y_batch[i].item()] += 1
                if logits_hard[i] == y_batch[i]:
                    pred[logits_hard[i].item()] += 1
        print(pred, gt, pred/gt)
        y_true, y_pred = gt_all.numpy().astype(np.uint8), pred_all.numpy().astype(np.uint8)
        test_accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        auc = metrics.roc_auc_score(y_true, y_pred, average='macro')
    l.info(f"Test Accuracy: {test_accuracy.item():.3f}, f1:{f1:.3f}, precision:{p:.3f}, recall:{r:.3f}, auc:{auc:.3f}, \
        epoch:{max_epoch}")

if __name__ == '__main__':
    main()
