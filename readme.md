# Learning Image Demoireing from Unpaired Real Data

Pytorch implementation for "Learning Image Demoireing from Unpaired Real Data".

## Requirements

- python 3.7
- pytorch 1.9.0
- torchvision 0.11.3

## Taining from from scratch

### 1, data preparation

First, the training images should be cropped into patches.
Then, sort the cropped patches according their complexity.

For fhdmi, go to the directory: './data_script/fhdmi/', run
```shell
python split_patches_train.py
python get_im_score_train.py
```

For uhdm, go to the directory: './data_script/uhdm/', run
```shell
python copy.py # this file is for copying UHDM train images to a single directory
python split_patches_train.py
python get_im_score_train.py
```
Note that set the data path in these python files:
```shell
Lines23-26 in split_patches_train.py
Lines19-25 in get_im_score_train.py
```

We suggest the patches of each class should be organized as the following way:
```shell
./data/fhdmi_class/
      train/
        class1/
        class2/
        class3/
        class4/
```


### 2, moire generation network training

go to the directory: './moire_syn', run:
```shell
python train_generate_mo.py 
        --traindata_path ./data/fhdmi_class/train/class1 
        --savefilename ./data/fhdmi_pacth384_fakemodel_class1 
        --dataset fhdmi --patch_size 384
```
We should run the above command for each class (the class of patches) by set the 'traindata_path'. 

We suggest the trained moire generation model of each class can be organized as the following way:
```shell
./models/fhdmi_fake_384/
      fhdmi_fake_class1_384/model.pth
      fhdmi_fake_class2_384/model.pth
      fhdmi_fake_class3_384/model.pth
      fhdmi_fake_class4_384/model.pth
```

### 3, demoire network training

Before demoire network training, we should obtain the threshold of the adaptive noise.
Run:
```shell
python get_sort.py --dataset fhdmi --patch_size 384 --class_index 1
```
Note that set the data path in get_sort.py:
```shell
Lines24/29, Lines59-105
```

Then, according to the desired threshold, put them based on the datasets (uhdm or fhdmi) and patch size (192 or 384 or 768) into the in the 
```shell
Lines 384-443 of train_oneModel.py
```


Finally, we can run the script in './run/' to training the demoire network.
```shell
run/mbcnn.sh
```
Also, please set the data path in these .sh file. 
