import argparse
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import time
from sklearn import metrics
import numpy as np
from dataset.alb_dataset import Tumor_dataset, Tumor_dataset_val, get_loader
import pandas as pd
from open_clip import create_model_from_pretrained, get_tokenizer
import random 
from sklearn.cluster import KMeans
from evaluate_util import hungarian_evaluate

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cls_recall(args, pred_array, target):
    pred, gt = np.zeros((args.num_class,)), np.zeros((args.num_class,))
    for i in range(len(target)):
        gt[target[i]] += 1
        if target[i]==pred_array[i]:
            pred[target[i]] += 1
    print(pred/gt, pred, gt)
    return pred/gt

def cluster_filter(args, feature_all, y_preds):
    cluster_learner = KMeans(n_clusters=args.num_class, init='k-means++', n_init='auto')
    cluster_learner.fit(feature_all)
    cluster_idxs = cluster_learner.predict(feature_all)
    cluster_pred = np.array(cluster_idxs, dtype=np.uint8)
    hungarian_results = hungarian_evaluate(torch.tensor(y_preds).cpu(), torch.tensor(cluster_pred).cpu())
    reordered_preds = hungarian_results['reordered_preds']
    return reordered_preds.numpy()==y_preds, reordered_preds

def get_files(data_root):
    new_file = []
    img_names = os.listdir(data_root+'images')
    for img_name in img_names:
        image_root = data_root+'images/'+img_name
        label_root = data_root+'labels/'+img_name
        label_img = np.array(Image.open(label_root))
        if label_img.max() > 0:
            label = 1
        else:
            label = 0
        new_sample = {'img': image_root, 'label': label}
        new_file.append(new_sample)
    return new_file

def get_arguments():
    parser = argparse.ArgumentParser(
        description="xxxx Pytorch implementation")
    parser.add_argument("--num_class", type=int, default=2, help="Train class num")
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--crop_size", default=224)
    parser.add_argument("--gpu", nargs="+", type=int)
    parser.add_argument("--batch_size", type=int, default=512, help="Train batch size")
    parser.add_argument("--num_workers", default=6)
    return parser.parse_args()

if __name__ == "__main__":
    seed_torch(42)
    args = get_arguments()
    torch.cuda.set_device(args.gpu[0])

    # dataset
    # train_data_root = '/home/ubuntu/data/lanfz/datasets/RINGS/train-all-patch/'
    train_data_root = '/home/ubuntu/data/lanfz/datasets/RINGS/train-100-patch/'
    val_data_root = '/home/ubuntu/data/lanfz/datasets/RINGS/val-patch/'
    test_data_root = '/home/ubuntu/data/lanfz/datasets/RINGS/test-patch/'
    train_files = get_files(train_data_root)
    val_files = get_files(val_data_root)
    test_files = get_files(test_data_root)
    val_files, test_files = val_files+test_files, val_files+test_files
    
    np.random.shuffle(train_files)
    print(len(train_files))
    train_dataset = Tumor_dataset(args, files=train_files)
    train_dataset_eval = Tumor_dataset_val(args, files=train_files)
    train_loader = get_loader(args, train_dataset, shuffle=False)
    train_eval_loader = get_loader(args, train_dataset_eval, shuffle=False)

    # get plip model   
    model = CLIPModel.from_pretrained("/home/ubuntu/data/lanfz/codes/CLIP-main/plip")
    processor = CLIPProcessor.from_pretrained("/home/ubuntu/data/lanfz/codes/CLIP-main/plip")

    model = model.cuda()
    model.eval()
    # model.train()

    t1 = time.time()

    text_prompt = ["An H&E image of normal tissue", "An H&E image of cancer tissue"]
    # text_prompt = ["An H&E image of healthy tissue", "An H&E image of cancer tissue"]
    inputs = processor(text=text_prompt, return_tensors="pt", padding=True)
    
    dropout_n = 30
    names = []
    with torch.no_grad():
        for j in range(dropout_n):
            pred, gt = np.zeros((args.num_class,)), np.zeros((args.num_class,))
            pred_all, gt_all, prob_all = torch.zeros((1, )), torch.zeros((1, )), torch.zeros((1, args.num_class))
            # embeddings = torch.zeros((1, 768))
            embeddings = torch.zeros((1, 512))
            threshold = 0
            for counter, sample in enumerate(train_loader):
                x_batch = sample['img'].cuda()
                y_batch = sample['cls_label'].cuda()
                batch_names = sample['img_name']

                # for transformer models
                inputs['pixel_values'] = x_batch
                for key in inputs.keys():
                    inputs[key] = inputs[key].cuda()
                outputs = model.forward(**inputs)
                # this is the image-text similarity score
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

                logits_hard = torch.argmax(probs, dim=1)
                pred_all = torch.cat((pred_all, logits_hard.cpu()), dim=0)
                gt_all = torch.cat((gt_all, y_batch.cpu()), dim=0)
                prob_all = torch.cat((prob_all, probs.cpu()), dim=0)
                names += batch_names
                # embeddings = torch.cat((embeddings, outputs.vision_model_output.pooler_output.cpu()), dim=0)
                embeddings = torch.cat((embeddings, outputs.image_embeds.cpu()), dim=0)
                # embeddings = torch.cat((embeddings, image_features.cpu()), dim=0)

                for i in range(logits_hard.shape[0]):
                    gt[y_batch[i].item()] += 1
                    if logits_hard[i] == y_batch[i]:
                        pred[logits_hard[i].item()] += 1

                if counter == 0:
                    print(batch_names[0])
            if j==0:
                probs_n = prob_all.unsqueeze(2)
                pred_n = pred_all.unsqueeze(1)
            else:
                probs_n = torch.cat([probs_n, prob_all.unsqueeze(2)], dim=2)
                pred_n = torch.cat([pred_n, pred_all.unsqueeze(1)], dim=1)
    print(probs_n.shape)
    print(pred/gt, (pred/gt).mean())

    pred_all, gt_all, probs_n, pred_n = pred_all[1:], gt_all[1:], probs_n[1:], pred_n[1:]
    embeddings = embeddings[1:]
    names = np.array(names)
    y_true, y_pred = gt_all.numpy().astype(np.uint8), pred_all.numpy().astype(np.uint8)

    # Here use entropy equals to zero
    pred_n_sum = pred_n.sum(1)

    # here use uncertainty to select x% most reliable samples
    pred_n_avg = pred_n_sum/dropout_n
    pred_n_prob = torch.cat([pred_n_avg.unsqueeze(1), 1-pred_n_avg.unsqueeze(1)], dim=1)
    pred_entropy = -torch.sum(torch.log(pred_n_prob+1e-6)*pred_n_prob, dim=1)
    print(pred_entropy.shape)
    idx_un = pred_entropy.sort()[1][:int(0.3*pred_entropy.shape[0])].cpu()

    with torch.no_grad():
        pred, gt = np.zeros((args.num_class,)), np.zeros((args.num_class,))
        pred_all, gt_all, prob_all = torch.zeros((1, )), torch.zeros((1, )), torch.zeros((1, args.num_class))
        # embeddings = torch.zeros((1, 768))
        embeddings = torch.zeros((1, 512))
        names = []
        for counter, sample in enumerate(train_eval_loader):
            x_batch = sample['img'].cuda()
            y_batch = sample['cls_label'].cuda()
            batch_names = sample['img_name']

            # for transformer models
            inputs['pixel_values'] = x_batch
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
            outputs = model.forward(**inputs)
            # this is the image-text similarity score
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

            logits_hard = torch.argmax(probs, dim=1)
            pred_all = torch.cat((pred_all, logits_hard.cpu()), dim=0)
            gt_all = torch.cat((gt_all, y_batch.cpu()), dim=0)
            prob_all = torch.cat((prob_all, probs.cpu()), dim=0)
            names += batch_names
            # embeddings = torch.cat((embeddings, outputs.vision_model_output.pooler_output.cpu()), dim=0)
            embeddings = torch.cat((embeddings, outputs.image_embeds.cpu()), dim=0)
            # embeddings = torch.cat((embeddings, image_features.cpu()), dim=0)

            for i in range(logits_hard.shape[0]):
                gt[y_batch[i].item()] += 1
                if logits_hard[i] == y_batch[i]:
                    pred[logits_hard[i].item()] += 1
    pred_all, gt_all, embeddings = pred_all[1:], gt_all[1:], embeddings[1:]
    y_true, y_pred = gt_all.numpy().astype(np.uint8), pred_all.numpy().astype(np.uint8)

    # use idx_un
    y_pred, y_true, embeddings, names = y_pred[idx_un], y_true[idx_un], embeddings[idx_un], np.array(names)[idx_un]

    test_accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = metrics.recall_score(y_true, y_pred, average='macro')
    auc = metrics.roc_auc_score(y_true, y_pred)
    print(len(y_true), len(names), pred, gt)
    print(f"Test Accuracy: {test_accuracy.item()}, f1:{f1}, precision:{p}, recall:{r}, auc:{auc}")

    idx, _ = cluster_filter(args, embeddings.numpy(), y_pred)
    y_pred, y_true, names = y_pred[idx], y_true[idx], names[idx]

    test_accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = metrics.recall_score(y_true, y_pred, average='macro')
    auc = metrics.roc_auc_score(y_true, y_pred)
    print(len(y_true), len(names))
    cls_recall(args, y_pred, y_true)
    print(f"Test Accuracy: {test_accuracy.item()}, f1:{f1}, precision:{p}, recall:{r}, auc:{auc}")

    # write pandas
    data_df = pd.DataFrame()
    data_df['image_path'] = names
    data_df['pseudo_label'] = y_pred
    data_df['true_label'] = y_true
    data_df.to_csv('pseudo_data/HPH_un.csv', index=False)
    t2 = time.time()

