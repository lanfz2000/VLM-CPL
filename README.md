# VLM-CPL

### Overall Framework
There are three steps to train the classification network.
![overall](https://github.com/lanfz2000/VLM-CPL/blob/main/overall-1.png)

### Data prepare
Download the HPH dataset [here](https://data.mendeley.com/datasets/h8bdwrtnr5/1)

### Training process

First, use the on-the-shelf VLM for zero-shot inference with our proposed method to filter out noisy samples on the training set.
```
python zero_shot_HPH_un.py --gpu 0
```
Second, after obtaining high-quality pseudo-labels, you can train a classification network.
```
python train_pseudo.py --gpu 0
```
