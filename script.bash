#!/bin/bash

# abort on error
set -e


### ResNet50

device=1
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
        --deterministic --dataset $dataset --trial_seed $seed --drop_mode filter --drop_activation False
    done
done

### ViT

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
        --deterministic --dataset $dataset --trial_seed $seed --drop_mode point --drop_activation False
    done
done