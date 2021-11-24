# Generalizing Adversarial Training to Composite Semantic Perturbations

## Scripts for running the code
### Evaluate robust accuracy / attack success rate of the model
#### Cifar-10
```shell
python3 evaluate_cifar10_robustness.py --arch wideresnet \
        --checkpoint PATH_TO_MODEL.pt --stat-dict CHECKPOINT_MODEL_TYPE \
        --message MESSAGE_TO_PRINT_IN_CSV \
        --batch_size BATCH_SIZE --output RESULT.csv \
        "NoAttack()" \
        "CompositeAttack(model, enabled_attack=(0,), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(1,), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(2,), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(3,), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(4,), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(5,), order_schedule='fixed', inner_iter_num=20, attack_power='strong')" \
        "AutoLinfAttack(model, 'cifar', bound=8/255)"
```
If the numbers of enabled attack are more than one, the batch size should be set to one making each image could update its own attack order.

```shell
python3 evaluate_cifar10_robustness.py --arch wideresnet \
        --checkpoint PATH_TO_MODEL.pt --stat-dict CHECKPOINT_MODEL_TYPE \
        --message MESSAGE_TO_PRINT_IN_CSV \
        --batch_size BATCH_SIZE --output RESULT.csv \
        "CompositeAttack(model, enabled_attack=(0,5), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(1,5), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(2,5), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(3,5), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(4,5), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
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
        --batch_size BATCH_SIZE --output RESULT.csv \
        "NoAttack()" \
        "CompositeAttack(model, enabled_attack=(0,), order_schedule='fixed', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(1,), order_schedule='fixed', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(2,), order_schedule='fixed', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(3,), order_schedule='fixed', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(4,), order_schedule='fixed', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(5,), order_schedule='fixed', inner_iter_num=20, attack_power='im_strong')" \
        "AutoLinfAttack(model, 'imagenet100', bound=4/255)"
```

### Generate Adversarial Examples and its differences
- `--dataset`: 'cifar' or 'imagenet'.
- `--robust_num`: print the unsuccessful examples for the first *robust_num* attacks.
```shell
python3 generate_adv_examples.py --arch resnet50 \
        --checkpoint PATH_TO_MODEL_FOR_RESUMING.pt --stat-dict CHECKPOINT_MODEL_TYPE \
        --layout horizontal_alternate \
        --dataset imagenet --robust_num 0 \
        --batch_size BATCH_SIZE --only_successful --shuffle --output ADV_EXPS.pdf \
        "CompositeAttack(model, enabled_attack=(0,), order_schedule='fixed', inner_iter_num=10, attack_power='im_tough')" \
        "CompositeAttack(model, enabled_attack=(1,), order_schedule='fixed', inner_iter_num=10, attack_power='im_tough')" \
        "CompositeAttack(model, enabled_attack=(2,), order_schedule='fixed', inner_iter_num=10, attack_power='im_tough')" \
        "CompositeAttack(model, enabled_attack=(3,), order_schedule='fixed', inner_iter_num=10, attack_power='im_tough')" \
        "CompositeAttack(model, enabled_attack=(4,), order_schedule='fixed', inner_iter_num=10, attack_power='im_tough')" \
        "CompositeAttack(model, enabled_attack=(5,), order_schedule='fixed', inner_iter_num=10, attack_power='im_tough')"
```

### Adversarial Training
#### Cifar-10
##### Single Node, MultiGPUs, Single-Processing, Multi-Threading (TRADES Loss)
```shell
python3 train_cifar10.py \
        --batch-size BATCH_SIZE --epochs 150 --arch wideresnet \
        --checkpoint PATH_TO_MODEL_FOR_RESUMING.pt --stat-dict CHECKPOINT_MODEL_TYPE \
        --mode adv_train_trades --order random --enable 0,1,2,3,4,5 --power strong \
        --model-dir DIR_TO_SAVE_EPOCH/ \
        --log_filename TRAINING_LOG.csv
```

##### Distributed, MultiGPUs, Multi-Processing (Madry's Loss)
```shell

python3 train_cifar10.py \
        --dist-backend 'nccl' --multiprocessing-distributed \
        --batch-size BATCH_SIZE --epochs 150 --arch wideresnet \
        --checkpoint PATH_TO_MODEL_FOR_RESUMING.pt --stat-dict CHECKPOINT_MODEL_TYPE \
        --mode adv_train_madry --order random --enable 0,1,2,3,4,5 --power strong \
        --model-dir DIR_TO_SAVE_EPOCH/ \
        --log_filename TRAINING_LOG.csv
```

#### ImageNet
Applying Madry's Loss with Finetuning on L-inf robust model.
```shell
python3 train_imagenet.py \
        --dist-backend 'nccl' --multiprocessing-distributed \
        --batch-size BATCH_SIZE  --epochs 150 --arch resnet50 \
        --checkpoint PATH_TO_MODEL_FOR_RESUMING.pt --stat-dict TYPE_OF_CHECKPOINT \
        --mode adv_train_madry --order random --enable 0,1,2,3,4,5 --power im_strong \
        --model-dir DIR_TO_SAVE_EPOCH \
        --log_filename TRAINING_LOG.csv
```

### Generate Loss Landscape
Look for the `loss_landscape_analysis_ImageNet.ipynb` notebook for more details.


## Ack
<span style="color:#888;">This repo is benefit from [perceptual-advex](https://github.com/cassidylaidlaw/perceptual-advex/) 
[Laidlaw *et al.*], [TRADES](https://github.com/yaodongyu/TRADES) [Zhang *et al.*], 
and [robustness](https://github.com/MadryLab/robustness) [MadryLab].</span>
