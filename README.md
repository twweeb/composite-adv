# Generalizing Adversarial Training to Composite Semantic Perturbations

## Scripts for running the code
### Evaluate robust accuracy / attack success rate of the model
#### Cifar-10
```shell
python3 evaluate_model_robustness.py --arch trades-wrn \
        --checkpoint PATH_TO_MODEL.pt --stat-dict gat \
        --message MESSAGE_TO_PRINT_IN_CSV \
        --batch_size 256 --output RESULT.csv \
        "NoAttack()" \
        "CompositeAttack(model, enabled_attack=(0,), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(1,), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(2,), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(3,), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(4,), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(5,), order_schedule='fixed', inner_iter_num=5, attack_power='strong')"
```
If the numbers of enabled attack are more than one, the batch size should be set to one making each image could update its own attack order.

```shell
python3 evaluate_cifar10_robustness.py --arch wideresnet \
        --checkpoint PATH_TO_MODEL.pt --stat-dict gat \
        --message MESSAGE_TO_PRINT_IN_CSV \
        --batch_size 1 --output RESULT.csv \
        "CompositeAttack(model, enabled_attack=(0,5), order_schedule='random', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(1,5), order_schedule='random', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(2,5), order_schedule='random', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(3,5), order_schedule='random', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(4,5), order_schedule='random', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(0,1,5), order_schedule='random', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(0,1,5), order_schedule='scheduled', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(3,4,5), order_schedule='random', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(3,4,5), order_schedule='scheduled', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(0,2,5), order_schedule='random', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(0,2,5), order_schedule='scheduled', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(0,1,2,3,4), order_schedule='random', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(0,1,2,3,4), order_schedule='scheduled', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(0,1,2,3,4,5), order_schedule='random', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(0,1,2,3,4,5), order_schedule='scheduled', inner_iter_num=5, attack_power='strong')"
```

#### ImageNet
```shell
python3 evaluate_imagenet_robustness.py --arch resnet50 \
        --checkpoint PATH_TO_MODEL.pt \
        --message MESSAGE_TO_PRINT_IN_CSV \
        --batch_size 192 --output RESULT.csv \
        "NoAttack()" \
        "CompositeAttack(model, enabled_attack=(0,), order_schedule='fixed', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(1,), order_schedule='fixed', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(2,), order_schedule='fixed', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(3,), order_schedule='fixed', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(4,), order_schedule='fixed', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(5,), order_schedule='fixed', inner_iter_num=5, attack_power='im_strong')"
```

### Generate Adversarial Examples and its differences

```shell
python3 generate_adv_examples.py --arch resnet50 \
        --layout horizontal_alternate \
        --batch_size 256 --only_successful --shuffle --output ADV_EXPS.pdf \
        "CompositeAttack(model, enabled_attack=(0,), order_schedule='fixed', inner_iter_num=10, attack_power='im_tough')" \
        "CompositeAttack(model, enabled_attack=(1,), order_schedule='fixed', inner_iter_num=10, attack_power='im_tough')" \
        "CompositeAttack(model, enabled_attack=(2,), order_schedule='fixed', inner_iter_num=10, attack_power='im_tough')" \
        "CompositeAttack(model, enabled_attack=(3,), order_schedule='fixed', inner_iter_num=10, attack_power='im_tough')" \
        "CompositeAttack(model, enabled_attack=(4,), order_schedule='fixed', inner_iter_num=10, attack_power='im_tough')" \
        "CompositeAttack(model, enabled_attack=(5,), order_schedule='fixed', inner_iter_num=10, attack_power='im_tough')"
```

### Generate Loss Landscape

### Adversarial Training
#### Cifar-10
TRADES Loss
```shell
python3 train_trades_cifar10.py \
        --batch-size 1024 --epochs 50 --arch wideresnet \
        --checkpoint PATH_TO_MODEL_FOR_RESUMING.pt --stat-dict CHECKPOINT_MODEL_TYPE \
        --mode adv_train_trades --order random --enable 0,1,2,3,4,5 --power strong --dist comp \
        --model-dir DIR_TO_SAVE_EPOCH/ \
        --log_filename TRAINING_LOG.csv
```

Normal Loss (Distributed, MultiGPUs, Single Node)
```shell
python3 train_trades_cifar10.py \
        --dist-url 'tcp://127.0.0.1:9527' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
        --batch-size 4096 --epochs 30 --arch wideresnet \
        --checkpoint PATH_TO_MODEL_FOR_RESUMING.pt --stat-dict CHECKPOINT_MODEL_TYPE \
        --mode adv_train --order random --enable 0,1,2,3,4,5 --power strong \
        --model-dir DIR_TO_SAVE_EPOCH/ \
        --log_filename TRAINING_LOG.csv
```

#### ImageNet
Using Normal Loss (Distributed, MultiGPUs, Single Node)
```shell
python3 finetune_madry_imagenet.py \
        --dist-url 'tcp://127.0.0.1:9527' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
        --batch-size 1024 --epochs 50 --arch resnet50 \
        --checkpoint PATH_TO_MODEL_FOR_RESUMING.pt --stat-dict TYPE_OF_CHECKPOINT \
        --mode adv_train --order random --enable 0,1,2,3,4,5 --power im_strong \
        --model-dir DIR_TO_SAVE_EPOCH \
        --log_filename TRAINING_LOG.csv
```

## Ack
<span style="color:#888;">This repo is benefit from [perceptual-advex](https://github.com/cassidylaidlaw/perceptual-advex/) 
[Laidlaw *et al.*], [TRADES](https://github.com/yaodongyu/TRADES) [Zhang *et al.*], 
and [robustness](https://github.com/MadryLab/robustness) [MadryLab].</span>


 

