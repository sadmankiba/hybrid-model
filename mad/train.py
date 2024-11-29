from torch.utils.data import DataLoader


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
    
    
if __name__ == '__main__':
    args = parse_args()
    
    config = MADConfig()
    config.update_from_kwargs(vars(args))
    
    model = Model() 
    
    train(model=model, config=config)