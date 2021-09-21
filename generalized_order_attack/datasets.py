import torch
import os
import numpy as np

from torchvision.datasets import CIFAR10

from robustness.datasets import CIFAR, DATASETS, DataSet
from robustness import data_augmentation
from torchvision.datasets.vision import VisionDataset

class CIFAR10C(CIFAR):
    """
    CIFAR-10-C from https://github.com/hendrycks/robustness.
    """

    def __init__(
        self,
        data_path,
        corruption_type: str = 'gaussian_noise',
        severity: int = 1,
        **kwargs,
    ):
        class CustomCIFAR10(CIFAR10):
            def __init__(self, root, train=True, transform=None,
                         target_transform=None, download=False):
                VisionDataset.__init__(self, root, transform=transform,
                                       target_transform=target_transform)

                if train:
                    raise NotImplementedError(
                        'No train dataset for CIFAR-10-C')
                if download and not os.path.exists(root):
                    raise NotImplementedError(
                        'Downloading CIFAR-10-C has not been implemented')

                all_data = np.load(
                    os.path.join(root, f'{corruption_type}.npy'))
                all_labels = np.load(os.path.join(root, f'labels.npy'))

                severity_slice = slice(
                    (severity - 1) * 10000,
                    severity * 10000,
                )

                self.data = all_data[severity_slice]
                self.targets = all_labels[severity_slice]

        DataSet.__init__(
            self,
            'cifar10c',
            data_path,
            num_classes=10,
            mean=torch.tensor([0.4914, 0.4822, 0.4465]),
            std=torch.tensor([0.2023, 0.1994, 0.2010]),
            custom_class=CustomCIFAR10,
            label_mapping=None,
            transform_train=data_augmentation.TRAIN_TRANSFORMS_DEFAULT(32),
            transform_test=data_augmentation.TEST_TRANSFORMS_DEFAULT(32)
        )


DATASETS['cifar10c'] = CIFAR10C