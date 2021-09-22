# Generalizing Adversarial Training to Composite Semantic Perturbations

## Scripts for running the code
### Evaluate robust accuracy / attack success rate of the model
- Cifar-10
```shell
python3 evaluate_model_robustness.py --arch trades-wrn \
        --checkpoint PATH_TO_MODEL.pt \
        --message MESSAGE_TO_PRINT_IN_CSV \
        --batch_size 50 --output RESULT.csv \
        "NoAttack()" \
        "CompositeAttack(model, enabled_attack=(0,), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(1,), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(2,), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(3,), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(4,), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(5,), order_schedule='fixed', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(5,0), order_schedule='random', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(5,1), order_schedule='random', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(5,2), order_schedule='random', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(5,3), order_schedule='random', inner_iter_num=5, attack_power='strong')" \
        "CompositeAttack(model, enabled_attack=(5,4), order_schedule='random', inner_iter_num=5, attack_power='strong')" \
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

- ImageNet
```shell
python3 evaluate_imagenet_robustness.py --arch resnet50 \
        --checkpoint PATH_TO_MODEL.pt \
        --message MESSAGE_TO_PRINT_IN_CSV \
        --batch_size 192 --output RESULT.csv \
        "NoAttack()" \
        "CompositeAttack(model, enabled_attack=(0,1,5), order_schedule='random', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(0,1,5), order_schedule='scheduled', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(3,4,5), order_schedule='random', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(3,4,5), order_schedule='scheduled', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(0,2,5), order_schedule='random', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(0,2,5), order_schedule='scheduled', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(0,1,2,3,4), order_schedule='random', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(0,1,2,3,4), order_schedule='scheduled', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(0,1,2,3,4,5), order_schedule='random', inner_iter_num=5, attack_power='im_strong')" \
        "CompositeAttack(model, enabled_attack=(0,1,2,3,4,5), order_schedule='scheduled', inner_iter_num=5, attack_power='im_strong')"
```

### Generate Adversarial Examples and its differences

```shell
python3 generate_adv_examples.py --arch resnet50 \
        --layout horizontal_alternate \
        --batch_size 256 --only_successful --shuffle --output ADV_EXPS.pdf \
        "CompositeAttack(model, enabled_attack=(0,), order_schedule='random', inner_iter_num=10, attack_power='im_tough')" \
        "CompositeAttack(model, enabled_attack=(1,), order_schedule='random', inner_iter_num=10, attack_power='im_tough')" \
        "CompositeAttack(model, enabled_attack=(2,), order_schedule='random', inner_iter_num=10, attack_power='im_tough')" \
        "CompositeAttack(model, enabled_attack=(3,), order_schedule='random', inner_iter_num=10, attack_power='im_tough')" \
        "CompositeAttack(model, enabled_attack=(4,), order_schedule='random', inner_iter_num=10, attack_power='im_tough')" \
        "CompositeAttack(model, enabled_attack=(5,), order_schedule='random', inner_iter_num=10, attack_power='im_tough')"
```

### Generate Loss Landscape

### Adversarial Training
- Cifar-10
```shell
python3 train_trades_cifar10_nodist.py --batch-size 128 --epochs 50 \
        --model-dir DIR_TO_SAVE_EPOCH \
        --log_filename TRAINING_LOG.csv \
        --order random --enable 0,1,2,3,4,5 --power strong \
        --dist comp \
        --model_path PATH_TO_MODEL_FOR_RESUMING.pt
```

- ImageNet
```shell
python3 finetune_madry_imagenet.py --batch-size 256 --epochs 50 \
        --arch resnet50 \
        --checkpoint PATH_TO_MODEL_FOR_RESUMING.pt --stat-dict TYPE_OF_CHECKPOINT \
        --mode adv_train --order random --enable 0,1,2,3,4,5 --power im_strong --dist comp \
        --model-dir DIR_TO_SAVE_EPOCH \
        --log_filename TRAINING_LOG.csv
```

## Ack
<span style="color:#888;">This repo is benefit from [perceptual-advex](https://github.com/cassidylaidlaw/perceptual-advex/) 
[Laidlaw *et al.*], [TRADES](https://github.com/yaodongyu/TRADES) [Zhang *et al.*], 
and [robustness](https://github.com/MadryLab/robustness) [MadryLab].</span>


 

