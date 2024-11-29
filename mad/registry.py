from instances import (
    generate_in_context_recall_instance,
    generate_noisy_in_context_recall_instance,
    generate_fuzzy_in_context_recall_instance,
    generate_memorization_instance,
    generate_compression_instance,
    generate_selective_copying_instance
)


task_registry = {
    'in-context-recall': {
        'instance_fn': generate_in_context_recall_instance,
        'shorthand': 'CR'
        
    },
    'noisy-in-context-recall': {
        'instance_fn': generate_noisy_in_context_recall_instance,
        'shorthand': 'NR'
    },
    'fuzzy-in-context-recall': {
        'instance_fn': generate_fuzzy_in_context_recall_instance,
        'shorthand': 'FR'
    },
    'memorization': {
        'instance_fn': generate_memorization_instance,
        'shorthand': 'M'
    },
    'compression': {
        'instance_fn': generate_compression_instance,
        'shorthand': 'C'
    },
    'selective-copying': {
        'instance_fn': generate_selective_copying_instance,
        'shorthand': 'SC'
    },
}