import os
from .registry import task_registry

def is_num(v) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False

def is_bool(v) -> bool:
    return v in {'True', 'False'} or isinstance(v, bool)

# Mapping of key names to their shorthands used in paths:
KEY_TO_SHORTHAND = {
    'task': 't',
    'vocab_size': 'vs',
    'seq_len': 'sl',
    'num_train_examples': 'ntr',
    'num_test_examples': 'nte',
    'k_motif_size': 'km',
    'v_motif_size': 'vm',
    'frac_noise': 'fn',
    'noise_vocab_size': 'nvs',
    'multi_query': 'mq',
    'num_tokens_to_copy': 'ntc',
    'seed': 's',
    'dim': 'd',
    'layers': 'lyr',
    'lr': 'lr',
    'weight_decay': 'wd',
    'epochs': 'e',
    'batch_size': 'bs',
    'optimizer': 'opt',
    'scheduler': 'sch'
}
# Mapping shorthands back to their key names:
SHORTHAND_TO_KEY = {v: k for k,v in KEY_TO_SHORTHAND.items()}

def make_dataset_path(mad_config, **kwargs):
    """Make a dataset path from MADConfig and additional kwargs."""
    if mad_config.task in task_registry:
        task = task_registry[mad_config.task]["shorthand"]
    else:
        task = mad_config.task
    path = f't-{task}_'
    for k in [
        'vocab_size',
        'seq_len',
        'num_train_examples',
        'num_test_examples',
        'k_motif_size',
        'v_motif_size',
        'multi_query',
        'frac_noise',
        'noise_vocab_size',
        'num_tokens_to_copy',
        'seed'
    ]:
        v = getattr(mad_config, k)
        if v is not None:
            if is_num(v):
                v = str(v).replace('.', '#')
            if is_bool(v):
                v = int(bool(v))
        path += f'{KEY_TO_SHORTHAND[k]}-{v}_'
    
    for k,v in kwargs.items():
        path += f'{k}-{v}_'
    
    path = path[:-1] # exclude last '_"
    return os.path.join(mad_config.data_path, path)