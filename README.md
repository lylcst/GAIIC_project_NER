# 代码说明

## 环境配置

本实验使用的python版本为3.8, cuda版本10.0其他主要的第三方库包括:

- pytorch==1.10.1
- transformers=4.17.0
- ark-nlp==0.0.8

## 数据



## 预训练模型

使用了NeZha预训练模型，可以通过GitHub链接https://github.com/lylcst/NeZha_Chinese_PyTorch 获得，
对code/models/modeling_nezha.py中的网络进行初始化

## 算法

### 整体思路介绍

模型结构：使用NeZha预训练模型 nezha-cn-base作为基础模型，使用GlobalPointer作为解码器进行解码。

1、加了warmup、梯度裁剪、weight_decay，调整超参数

2、使用了fgm对抗训练

3、使用了伪标签，用训练好的模型去标注了20000条无标签的数据(unlabeled_train_data.txt)，然后和训练集数据一起重新训练模型

4、加了rdrop略有提升

5、试过利用无标注的数据对nezha-cn-base做自监督的进一步预训练，但是还没有提升

### 网络结构

![image-20220423220155460](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20220423220155460.png)

### 损失函数

使用多标签类别的交叉熵损失函数：multilabel_categorical_crossentropy

### 其他说明

data/orther下放置了一个文件：fake_train_data_20000.txt，是用自己模型标注的伪标签数据集，用于训练阶段使用。

## 训练流程

```sh
sh train.sh
```
或
```sh
python code/train.py --train_data_file_path data/contest_data/train_data/train.txt \
                     --model_name data/pretrain_model/nezha-cn-base \
                     --model_save_dir data/model_data \
                     --learning_rate 2e-5 \
                     --num_epoches 6 \
                     --batch_size 32 \
                     --warmup_proportion 0.1 \
                     --gradient_accumulation_steps 1 \
                     --max_grad_norm 1.0 \
                     --weight_decay 0.01 \
                     --do_adv \
                     --do_fake_label \
                     --fake_train_data_path data/tmp_data \
                     --fake_train_data_name fake_train_data_20000.txt \
                     --rdrop \
                     --rdrop_rate 0.5 \
                     --cuda_device 0 \
                     --seed 42
```
参数列表:
|         FLAG             |    description    |     default     |
| :----------------------: | :---------------: | :--------------: |
|  --train_data_file_path |  训练数据文件train.txt的路径  |  None  |
| --model_name | 使用的模型初始化参数路径 | data/pretrain_model/nezha-cn-base |
|        --model_save_dir |  模型的保存根目录  |  ./data/model_data  |
|      --learning_rate |  模型训练的学习率  |  2e-5  |
| --num_epoches |  模型训练的epoch数  |  6  |
| --batch_size     |  模型训练的batch大小  |  32  |
| --warmup_proportion |  Proportion of training to perform linear learning rate warmup  |  0.1  |
| --gradient_accumulation_steps     |  Number of updates steps to accumulate before performing a backward/update pass  |  1  |
| --max_grad_norm |  Max gradient norm  | 1.0 |
|  --weight_decay |  Weight decay if we apply some  |  0.01  |
|      --do_adv   |  Whether to use fgm  |  False  |
|       --do_fake_label       |  是否使用伪标签训练  |  False  |
| --fake_train_data_path |  如果使用伪标签训练，此项设置伪标签文件的目录路径  |  ./data/orther  |
| --fake_train_data_name |  如果使用伪标签训练，此项设置伪标签文件的名字  |  None  |
|      --rdrop    |  是否使用rdrop  |  None  |
| --rdrop_rate | rdrop系数 | 0.5 |
| --cuda_device | 使用的GPU设备号 | 0 |
| --seed | 随机种子 | 42 |

## 测试流程

```sh
sh test.sh data/contest_data/preliminary_test_b/sample_per_line_preliminary_B.txt
```





