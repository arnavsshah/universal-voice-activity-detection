import torch

from src.scripts import train_vad, test_vad, predict_vad
from data import prepare_data, test_data

from config.config import load_config

def main(config):
    
    if config.task == 'prepare':
        prepare_data(**config)
    
    elif config.task == 'test_data':
        test_data(**config)

    elif config.task == 'run':
        if config.function == 'train':
            train_vad(**config)
        elif config.function == 'test':
            test_vad(**config)
        elif config.function == 'predict':
            predict_vad(**config)



if __name__ == '__main__':

    print(torch.cuda.is_available())
    
    # a = torch.ones(1).to("cuda")  # to avoid race condition, use cuda device at the start
    # print(a)

    config = load_config()
    main(config)
