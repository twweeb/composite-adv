# Composite Adversarial Perturbations

## Installation
Composite-adv can be downloaded as this GitHub repository, the code including training and evaluation phases.

- Install [Python 3](https://www.python.org/).
- Install as a package:
  ```shell
  pip install git+https://github.com/twweeb/composite-adv.git
  ```

## Pretrained Models
To be released.


## Usage
### Evaluate robust accuracy / attack success rate of the model
```shell
python evaluate_model.py \
       --arch resnet50 --checkpoint PATH_TO_MODEL \
       --dataset DATASET_NAME --dataset-path DATASET_PATH --input-normalized \
       --message MESSAGE_TO_PRINT_IN_CSV \
       --batch_size BATCH_SIZE --output RESULT.csv \
       "NoAttack()" \
       "CompositeAttack(model, enabled_attack=(0,), order_schedule='fixed', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(1,), order_schedule='fixed', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(2,), order_schedule='fixed', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(3,), order_schedule='fixed', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(4,), order_schedule='fixed', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(5,), order_schedule='fixed', inner_iter_num=20)" \
       "CompositeAttack(model, enabled_attack=(0,1,5), order_schedule='random', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(0,1,5), order_schedule='scheduled', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(3,4,5), order_schedule='random', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(3,4,5), order_schedule='scheduled', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(0,2,5), order_schedule='random', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(0,2,5), order_schedule='scheduled', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(0,1,2,3,4), order_schedule='random', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(0,1,2,3,4), order_schedule='scheduled', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(0,1,2,3,4,5), order_schedule='random', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(0,1,2,3,4,5), order_schedule='scheduled', inner_iter_num=10)" \
       "AutoLinfAttack(model, 'cifar', bound=8/255)"
```

### Adversarial Training
#### Cifar-10
##### Single Node, MultiGPUs, Single-Processing, Multi-Threading (TRADES Loss)
```shell
python3 train_cifar10.py \
        --batch-size BATCH_SIZE --epochs 150 --arch wideresnet \
        --checkpoint PATH_TO_MODEL_FOR_RESUMING.pt \
        --mode adv_train_trades --order random --enable 0,1,2,3,4,5 \
        --model-dir DIR_TO_SAVE_EPOCH/ \
        --log_filename TRAINING_LOG.csv
```

##### Distributed, MultiGPUs, Multi-Processing (Madry's Loss)
```shell
python3 train_cifar10.py \
        --dist-backend 'nccl' --multiprocessing-distributed \
        --batch-size BATCH_SIZE --epochs 150 --arch wideresnet \
        --checkpoint PATH_TO_MODEL_FOR_RESUMING.pt \
        --mode adv_train_madry --order random --enable 0,1,2,3,4,5 \
        --model-dir DIR_TO_SAVE_EPOCH/ \
        --log_filename TRAINING_LOG.csv
```

#### ImageNet
Applying Madry's Loss with Finetuning on Linf-robust model.
```shell
python3 train_imagenet.py \
        --dist-backend 'nccl' --multiprocessing-distributed \
        --batch-size BATCH_SIZE  --epochs 150 --arch resnet50 \
        --checkpoint PATH_TO_MODEL_FOR_RESUMING.pt --stat-dict TYPE_OF_CHECKPOINT \
        --mode adv_train_madry --order random --enable 0,1,2,3,4,5 \
        --model-dir DIR_TO_SAVE_EPOCH \
        --log_filename TRAINING_LOG.csv
```


## Citation
If you find this helpful and useful for your research, please cite our main paper as follows:

    @misc{tsai2022compositional,
      title={Towards Compositional Adversarial Robustness: Generalizing Adversarial Training to Composite Semantic Perturbations}, 
      author={Yun-Yun Tsai and Lei Hsiung and Pin-Yu Chen and Tsung-Yi Ho},
      year={2022},
      eprint={2202.04235},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
