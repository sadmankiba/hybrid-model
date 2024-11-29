import typing as tp
from typing import Optional

import torch
import numpy as np

def check_for_leakage(train_inputs, test_inputs):
    """Helper to check for data leakage between train and test sets."""
    train_set = set([" ".join(map(str, x)) for x in train_inputs.tolist()])
    test_set = set([" ".join(map(str, x)) for x in test_inputs.tolist()])
    frac_test_in_train = 1 - (len(test_set - train_set) / len(test_set))
    if frac_test_in_train > 0.001:
        print(
            "WARNING: Potential data leakage detected. " 
            f"{frac_test_in_train: 0.2f} of test examples are in the train set."
        )

def generate_data(
    instance_fn: tp.Callable,
    instance_fn_kwargs: tp.Dict,
    num_train_examples: int,
    num_test_examples: int,
    train_data_path: Optional[str] = None,
    test_data_path: Optional[str] = None,
) -> tp.Dict[str, tp.Any]:
    """
    
    Args:
        instance_fn: function to generate an instance of the task
        instance_fn_kwargs: kwargs to pass to instance_fn
        train_data_path: path to save train data
        test_data_path: path to save test data
        num_train_examples: number of training examples to generate
        num_test_examples: number of test examples to generate
    """
    # Generate train data
    train_instance_fn_kwargs = instance_fn_kwargs.copy()
    train_instance_fn_kwargs['is_training'] = True
    
    train_dataset = MadDataset(instance_fn=instance_fn, instance_fn_kwargs=train_instance_fn_kwargs)
    train_dataset.generate_data(num_examples=num_train_examples)
    if train_data_path is not None:
        train_dataset.save_data(train_data_path)
    
    # Generate test data
    test_instance_fn_kwargs = instance_fn_kwargs.copy()
    test_instance_fn_kwargs['is_training'] = False
    
    test_dataset = MadDataset(instance_fn=instance_fn, instance_fn_kwargs=test_instance_fn_kwargs)
    test_dataset.generate_data(num_examples=num_test_examples)
    if test_data_path is not None:
        test_dataset.save_data(test_data_path)
        
    # Check for data leakage
    check_for_leakage(train_dataset.inputs, test_dataset.inputs)
    
    return {'train': train_dataset, 'test': test_dataset}

class MadDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for a MAD task. 
    """
    
    def __init__(self, instance_fn, instance_fn_kwargs):
        self.instance_fn = instance_fn
        self.instance_fn_kwargs = instance_fn_kwargs
        self.inputs = None 
        self.targets = None
    
    def __getitem__(self, idx):
        assert self.inputs is not None, "no data generated yet"
        return self.inputs[idx], self.targets[idx]
    
    def __len__(self):
        assert self.inputs is not None, "no data generated yet"
        return len(self.inputs)
    
    def generate_data(self, num_examples):
        instances = [self.instance_fn(**self.instance_fn_kwargs) for _ in range(num_examples)]
        
        assert len(instances[-1]) == 2, "instance_fn must return a tuple of (input, target)"
        self.inputs, self.targets = [np.stack(i) for i in zip(*instances)]