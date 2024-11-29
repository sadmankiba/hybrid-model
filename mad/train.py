from torch.utils.data import DataLoader

from gen_data import generate_data


def train(model, config):
    data = generate_data(
        instance_fn=config.instance_fn,
        instance_fn_kwargs=config.instance_fn_kwargs,
        train_data_path=config.train_dataset_path,
        test_data_path=config.test_dataset_path,
        num_train_examples=config.num_train_examples,
        num_test_examples=config.num_test_examples,
    )
    train_dl = DataLoader(dataset=data['train'], batch_size=8, shuffle=True)
    
    test_dl = DataLoader(dataset=data['test'], batch_size=8, shuffle=False)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='in-context-recall')
    parser.add_argument('--vocab_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--frac_noise', type=float, default=0.0)
    parser.add_argument('--noise_vocab_size', type=int, default=0)
    parser.add_argument('--num_tokens_to_copy', type=int, default=0)
    parser.add_argument('--k_motif_size', type=int, default=1)
    parser.add_argument('--v_motif_size', type=int, default=1)
    parser.add_argument('--multi_query', type=bool, default=True)
    parser.add_argument('--num_train_examples', type=int, default=12_800)
    parser.add_argument('--num_test_examples', type=int, default=1_280)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--early_stop', type=bool, default=False)
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--target_ignore_index', type=int, default=-100)
    return parser.parse_args()