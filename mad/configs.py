import os
import typing as tp
from dataclasses import dataclass, fields

import numpy as np 
import torch.nn as nn

from .instances import generate_kv_map
from .paths import make_dataset_path
from .registry import task_registry


class BaseConfig:
    def update_from_kwargs(self, kwargs):
        """Update fields of the config with kwargs."""
        valid_keys = {field.name for field in fields(self)}
        for key, value in kwargs.items():
            if key in valid_keys:
                setattr(self, key, value)


@dataclass
class MADConfig(BaseConfig):
    """MAD configuration."""

    # task settings:
    mad_task: str = 'in-context-recall'
    vocab_size: int = 16
    seq_len: int = 128
    frac_noise: float = 0.0
    noise_vocab_size: int = 0
    num_tokens_to_copy: int = 0
    k_motif_size: int = 1
    v_motif_size: int = 1
    multi_query: bool = True
    num_train_examples: int = 12_800
    num_test_examples: int = 1_280

    # data settings:
    data_path: str = './data'

    # training settings:
    batch_size: int = 128
    epochs: float = 200
    lr: float = 5e-4
    weight_decay: float = 0.
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    min_lr: float = 1e-6
    early_stop: bool = False
    precision: str = 'bf16'
    log_interval: int = 0
    use_gpu: bool = False

    # misc:
    seed: int = 12345
    target_ignore_index: int = -100

    @property
    def instance_fn(self) -> tp.Callable:
        """returns function from registry used to generate an instance of the task"""
        if self.mad_task in task_registry:
            return task_registry[self.mad_task]['instance_fn']
        else:
            return None

    @property
    def instance_fn_kwargs(self) -> tp.Dict:
        """returns dict of all kwargs required to create an instance with self.instance_fn"""
        if self.mad_task == 'memorization':
            # We need to generate a kv_map for the memorization task.
            # As this mapping is fixed, we can generate it here,
            # avoiding that it is recreated every time a new task instance is created.
            if self.k_motif_size>1 or self.v_motif_size>1:
                print('/!\ setting {k,v}_motif_size to 1, as motifs>1 are not supported for the memorization task.')
            kv_map = generate_kv_map(
                vocab_size=self.vocab_size - 1, # also account for insert tokens
                k_motif_size=1,
                v_motif_size=1,
                seed=self.seed
            )
        else:
            kv_map = None
        return dict(
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            k_motif_size=self.k_motif_size,
            v_motif_size=self.v_motif_size,
            frac_noise=self.frac_noise,
            noise_vocab_size=self.noise_vocab_size,
            num_tokens_to_copy=self.num_tokens_to_copy,
            rng=np.random.default_rng(self.seed),
            multi_query=self.multi_query,
            kv_map=kv_map
        )

    @property
    def dataset_path(self):
        return make_dataset_path(self)

    @property
    def train_dataset_path(self) -> str:
        return os.path.join(self.dataset_path, 'train')

    @property
    def test_dataset_path(self) -> str:
        return os.path.join(self.dataset_path, 'test')

                
@dataclass
class ModelConfig(BaseConfig):
    """Model configuration for models"""
    vocab_size: int = 16
    num_layers: int = 4
    num_trans_layers: int = 4
    num_mamba_layers: int = 4
    hidden_size: int = 128
    num_heads: int = 4
    backbone: str = 'language-model'
    
    num_hybrid_blocks: int = 4
    proj_type: str = 'res'
    
    max_length: int = 1_280
    norm: nn.Module = nn.LayerNorm
    position_embeds: tp.Callable = None
    embed_drop_rate: float = 0.0
    num_labels:int = 2

@dataclass
class ImdbConfig(BaseConfig):
    """Model configuration for models"""
    #data_path: str = './data'
    # training settings:
    batch_size: int = 8
    epochs: float = 3
    lr: float = 5e-4
    weight_decay: float = 0.002
    #optimizer: str = 'adamw'
    #scheduler: str = 'cosine'
    #min_lr: float = 1e-6
    #early_stop: bool = False
    #precision: str = 'bf16'
    #log_interval: int = 0 
    log_interval:int = 50
    use_gpu: bool = True
    eval_size = 0
    train_size = 0
    device:int =0
