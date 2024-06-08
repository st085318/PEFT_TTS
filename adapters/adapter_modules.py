from dataclasses import dataclass, field
from typing import Any, Optional, List

import torch
from torch import nn as nn
from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil, LinearAdapter
from nemo.core.classes.mixins import adapter_mixin_strategies

from nemo.core.classes.mixins.adapter_mixin_strategies import ResidualAddAdapterStrategy, AbstractAdapterStrategy


class Adamar(nn.Module):
    def __init__(
        self,
        shape
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(shape))
    
    def forward(self, input):
        return input * self.weight

class IA3Adapter(nn.Module, AdapterModuleUtil):
    def __init__(
        self,
        in_features: int,
        n_head: int,
        d_head: int,
        adapter_strategy: adapter_mixin_strategies.AbstractAdapterStrategy = None,
    ):
        super().__init__()
        
        self.setup_adapter_strategy(adapter_strategy)

        self.module = Adamar(2 * n_head * d_head)
        self.n_head = n_head
        self.d_head = d_head

    def reset_parameters(self):
        self.module.weight = torch.ones(self.module.weight.shape)

    def forward(self, input):
        x = self.module(input)
        return x
    
    def adapter_freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

@dataclass
class IA3AdapterConfig:
    in_features: int
    n_head: int
    d_head: int
    adapter_strategy: Optional[Any] = field(
        default_factory=lambda: MultiplicationAdapterStrategyConfig()
    )
    _target_: str = "{0}.{1}".format(IA3Adapter.__module__, IA3Adapter.__name__)


class LoRaAdapter(nn.Module, AdapterModuleUtil):
    def __init__(
        self,
        in_features: int,
        dim: int,
        n_head: int,
        d_head: int,
        adapter_strategy: adapter_mixin_strategies.AbstractAdapterStrategy = None,
    ):
        super().__init__()

        self.module = nn.Sequential(
            nn.Linear(in_features, dim, bias=False),
            nn.Linear(dim, 2 * n_head * d_head, bias=False),
        )

        self.setup_adapter_strategy(adapter_strategy)

        self.reset_parameters()

    def reset_parameters(self):
        self.module[-1].weight.data *= 0

    def forward(self, input):
        return self.module(input)

    def adapter_freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

@dataclass
class LoRaAdapterConfig:
    in_features: int
    dim: int
    n_head: int
    d_head: int
    adapter_strategy: Optional[Any] = field(
        default_factory=lambda: SimpleAdapterStrategyConfig()
    )
    _target_: str = "{0}.{1}".format(LoRaAdapter.__module__, LoRaAdapter.__name__)


class SimpleAdapterStrategy(AbstractAdapterStrategy):
    def forward(self, input: torch.Tensor, adapter: torch.nn.Module, *, module: 'AdapterModuleMixin'):
        return self.compute_output(input, adapter, module=module)

    def compute_output(self, input, adapter: torch.nn.Module, *, module: 'AdapterModuleMixin'):
        out = adapter(input)
        return out


@dataclass
class SimpleAdapterStrategyConfig:
    _target_: str = "{0}.{1}".format(
        SimpleAdapterStrategy.__module__, SimpleAdapterStrategy.__name__
    ) 