# High-Rate Mixout: Revisiting Mixout for Robust Domain Generalization (WACV 2026)

Official PyTorch implementation of [High-Rate Mixout: Revisiting Mixout for Robust Domain Generalization](https://arxiv.org/abs/2510.06955).

Masih Aminbeidokhti, Heitor Rapela Medeiros, Srikanth Muralidharan, Eric Granger, Marco Pedersoli.


## Preparation

### Requirements

* Python 3.10
* PyTorch 2.7
* Torchvision 0.22

- Other dependencies are listed in [requirements.txt](requirements.txt).

To install requirements, run:

```sh
conda create -n hr_mixout python=3.10 -y
conda activate hr_mixout
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```

## Hardware

Most experiments can be reproduced using a single GPU with 24GB of memory.

## How to Run

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.

### ResNet50

```sh
device=0
dataset=DomainNet
steps=15000

lr=3e-5
wd=1e-4
drate=0.5
grate=0.7

for seed in 0
do
    for test_envs in 0
    do 
        name=R50_env${test_envs}_s${seed}_mixfilter_lr_${lr}_wd_${wd}_gp_${grate}
        python train_all.py $name --test_envs $test_envs \
        --device $device --lr $lr --weight_decay $wd --algorithm ERM \
        --rx True --steps $steps --resnet_dropout $drate --group_dropout $grate \
        --deterministic --dataset $dataset --trial_seed $seed \ 
        --drop_mode filter --drop_activation False
    done
done
```

### ViT

```sh
device=1
dataset=DomainNet
steps=15000

lr=3e-5
wd=1e-6
drate=0.0
grate=0.9

for seed in 0
do
    for test_envs in 0
    do
        name=ViT_env${test_envs}_s${seed}_mixout_lr_${lr}_wd_${wd}_gp_${grate}

        python train_all.py $name --test_envs $test_envs \
        --device $device --lr $lr --weight_decay $wd --algorithm ERM \
        --vit True --steps $steps --resnet_dropout $drate --group_dropout $grate \
        --deterministic --dataset $dataset --trial_seed $seed \ 
        --drop_mode point --drop_activation False
    done
done
```

## Acknowledgment

We thank the authors for the following repositories for code reference:
[[DomainBed]](https://github.com/facebookresearch/DomainBed/tree/main/domainbed), [[Mixout]](https://github.com/bloodwass/mixout), [[SWAD]](https://github.com/khanrc/swad/tree/main).