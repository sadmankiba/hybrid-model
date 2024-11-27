from hybrid.projector import Combiner, Splitter
from utils.params import count_parameters

def test_combiner_param_count():
    combiner = Combiner(10, 20)
    
    # params = 10 * 20 + 20 + 20 * 20 + 20 = 640
    total_params, trainable_params = count_parameters(combiner)
    assert total_params == 640
    assert trainable_params == 640
    

def test_splitter_param_count():
    splitter = Splitter(10, 20)
    
    # params = 20 * 10 + 10 + 20 * 20 + 20 = 630
    total_params, trainable_params = count_parameters(splitter)
    assert total_params == 630
    assert trainable_params == 630