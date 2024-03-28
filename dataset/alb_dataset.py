import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch.optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
Image.MAX_IMAGE_PIXELS = None
from torch.utils.data.sampler import Sampler
import itertools
from torch.utils.data import WeightedRandomSampler


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class Tumor_dataset(Dataset):
    def __init__(self, args, files):
        super().__init__()
        """
        args: params
        files: [{}, {}], "img", "label"
        """
        self.files = files
        # A.RandomRotate90, A.RandomGridShuffle(), A.ColorJitter(), A.GaussianBlur(), A.Sharpen(), A.RandomCrop(), A.Flip(), A.Affine(), A.ElasticTransform()
        
        self.train_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                A.RandomRotate90(),
                A.Flip(),
                # A.RandomGridShuffle(),
                # A.Affine(scale=(0.75, 1.25), rotate=[-45, 45]),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                # A.ElasticTransform(),
                # A.GaussianBlur(),
                # A.Sharpen(),
                # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                # A.RandomBrightnessContrast(p=0.2),
                A.ColorJitter(),
                A.Normalize(mean=[0.48145466,0.4578275,0.40821073],std=[0.26862954,0.26130258,0.27577711]),
                ToTensorV2(),
            ]
        )
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        label = cur_item['label']
        image = np.array(Image.open(image_path).convert('RGB'))
        transformed = self.train_transform(image=image)
        return {"img":transformed["image"], "cls_label":label, 'img_name':image_path}

    def __len__(self):
        return len(self.files)


class Tumor_dataset_pseudo(Dataset):
    def __init__(self, args, files):
        super().__init__()
        """
        args: params
        files: [{}, {}], "img", "label"
        """
        self.files = files
        # A.RandomRotate90, A.RandomGridShuffle(), A.ColorJitter(), A.GaussianBlur(), A.Sharpen(), A.RandomCrop(), A.Flip(), A.Affine(), A.ElasticTransform()
        
        self.train_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                A.RandomRotate90(),
                A.Flip(),
                # A.GaussianBlur(),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                # A.RandomBrightnessContrast(p=0.2),
                # A.ColorJitter(),
                A.Normalize(mean=[0.48145466,0.4578275,0.40821073],std=[0.26862954,0.26130258,0.27577711]),
                ToTensorV2(),
            ]
        )
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        p_label = cur_item['p_label']
        t_label = cur_item['t_label']
        image = np.array(Image.open(image_path).convert('RGB'))
        # print(np.max(mask))
        transformed = self.train_transform(image=image)

        return {"img":transformed["image"], "p_label":p_label, "t_label":t_label,'img_name':image_path, 'index':index}

    def __len__(self):
        return len(self.files)


class Tumor_dataset_two_weak(Dataset):
    def __init__(self, args, files):
        super().__init__()
        self.files = files
        self.train_transform_w1 = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
                A.RandomRotate90(),
                A.Flip(),
                # A.ColorJitter(),
                A.Normalize(mean=[0.48145466,0.4578275,0.40821073],std=[0.26862954,0.26130258,0.27577711]),
                ToTensorV2()
            ]
        )
        self.train_transform_w2 = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
                A.RandomRotate90(),
                A.Flip(),
                A.Normalize(mean=[0.48145466,0.4578275,0.40821073],std=[0.26862954,0.26130258,0.27577711]),
                ToTensorV2()
            ]
        )
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        image = Image.open(image_path).convert('RGB')
        cls_label = cur_item['label']
        image = np.array(image)
        image_w1 = self.train_transform_w1(image=image)["image"]
        image_w2 = self.train_transform_w2(image=image)["image"]

        return {"img1":image_w1, "img2":image_w2, "cls_label":cls_label, "img_name":image_path}

    def __len__(self):
        return len(self.files)

class Tumor_dataset_FM(Dataset):
    def __init__(self, args, files):
        super().__init__()
        self.files = files
        self.train_transform_w = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
                A.RandomRotate90(),
                A.Flip(),
                # A.ColorJitter(),
                A.Normalize(mean=[0.48145466,0.4578275,0.40821073],std=[0.26862954,0.26130258,0.27577711]),
                ToTensorV2()
            ]
        )
        self.train_transform_s = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
                A.RandomRotate90(),
                A.Flip(),
                A.ColorJitter(),
                A.Normalize(mean=[0.48145466,0.4578275,0.40821073],std=[0.26862954,0.26130258,0.27577711]),
                ToTensorV2()
            ]
        )
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        image = Image.open(image_path).convert('RGB')
        cls_label = cur_item['label']
        image = np.array(image)
        image_w = self.train_transform_w(image=image)["image"]
        image_s = self.train_transform_s(image=image)["image"]

        return {"img_w":image_w, "img_s":image_s, "cls_label":cls_label, "img_name":image_path}

    def __len__(self):
        return len(self.files)

class Tumor_dataset_val(Dataset):
    def __init__(self, args, files):
        super().__init__()
        self.files = files
        self.train_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.crop_size),
                A.Normalize(mean=[0.48145466,0.4578275,0.40821073],std=[0.26862954,0.26130258,0.27577711]),
                ToTensorV2(),
            ]
        )
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        mask_path = image_path.replace('images', 'labels')
        # mask_path = mask_path[:-4]+'_mask.png'
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))
        cls_label = 1 if np.max(mask) > 0 else 0
        transformed = self.train_transform(image=image, mask=mask)

        return {"img":transformed["image"], "cls_label":cls_label, "mask":transformed["mask"], 'img_name':image_path}

    def __len__(self):
        return len(self.files)


class Tumor_dataset_val_cls(Dataset):
    def __init__(self, args, files):
        super().__init__()
        self.files = files
        self.train_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.crop_size),
                A.Normalize(mean=[0.48145466,0.4578275,0.40821073],std=[0.26862954,0.26130258,0.27577711]),
                ToTensorV2(),
            ]
        )
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        label = cur_item['label']
        image = np.array(Image.open(image_path).convert('RGB'))
        transformed = self.train_transform(image=image)

        return {"img":transformed["image"], "cls_label":label, 'img_name':image_path}

    def __len__(self):
        return len(self.files)

class Tumor_dataset_val_cls_aug(Dataset):
    def __init__(self, args, files):
        super().__init__()
        self.files = files
        self.train_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.crop_size),
                A.RandomRotate90(),
                A.Flip(),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.48145466,0.4578275,0.40821073],std=[0.26862954,0.26130258,0.27577711]),
                ToTensorV2(),
            ]
        )
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        label = cur_item['label']
        image = np.array(Image.open(image_path).convert('RGB'))
        transformed = self.train_transform(image=image)

        return {"img":transformed["image"], "cls_label":label, 'img_name':image_path}

    def __len__(self):
        return len(self.files)


class Tumor_dataset_WSI(Dataset):
    def __init__(self, args, files):
        super().__init__()
        self.files = files
        self.train_transform = A.Compose(
            [
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        mask_path = cur_item['label']
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))
        transformed = self.train_transform(image=image, mask=mask)

        return {"img": transformed["image"], "label": transformed["mask"], 'img_name': image_path}

    def __len__(self):
        return len(self.files)

def get_loader(args, ds, shuffle=True, drop=False):
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=drop
    )
    return loader

def get_loader_WSI(ds):
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    return loader

def get_loader_resample(args, ds, weights):
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        sampler=WeightedRandomSampler(weights, len(weights))
    )
    return loader

def get_train_loader_ssl(args, ds, labeled_idxs, unlabeled_idxs):
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    train_loader = DataLoader(
        ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=False
    )
    return train_loader
