from mad.train import parse_args, train
from mad.configs import MADConfig


if __name__ == '__main__':
    args = parse_args()
    
    config = MADConfig()
    config.update_from_kwargs(vars(args))
    
    model = Model() 
    
    train(model=model, config=config)