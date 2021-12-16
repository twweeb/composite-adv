import torch.nn as nn
from .composite_attack import CompositeAttack


class AutoAttack(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()

        kwargs.setdefault('verbose', False)
        self.model = model
        self.kwargs = kwargs
        self.attack = None

    def forward(self, inputs, labels):
        # Necessary to initialize attack here because for parallelization
        # across multiple GPUs.
        if self.attack is None:
            try:
                import autoattack
            except ImportError:
                raise RuntimeError(
                    'Error: unable to import autoattack. Please install the '
                    'package by running '
                    '"pip install git+git://github.com/fra31/auto-attack#egg=autoattack".'
                )
            self.attack = autoattack.AutoAttack(
                self.model, device=inputs.device, **self.kwargs)

        return self.attack.run_standard_evaluation(inputs, labels)


class AutoLinfAttack(AutoAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'cifar': 8/255,
                'imagenet100': 8/255,
                'imagenet': 8/255,
                'bird_or_bicycle': 16/255,
            }[dataset_name]

        super().__init__(
            model,
            norm='Linf',
            eps=bound,
            **kwargs,
        )


class AutoL2Attack(AutoAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'cifar': 1,
                'imagenet100': 3,
                'imagenet': 3,
                'bird_or_bicycle': 10,
            }[dataset_name]

        super().__init__(
            model,
            norm='L2',
            eps=bound,
            **kwargs,
        )
