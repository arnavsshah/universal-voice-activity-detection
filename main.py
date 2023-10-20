import torch

# from src.scripts.train import train_vad
# # from src.data.prepare_data import prepare_data 

# from config.config import load_config

# def main(config):
    
#     # prepare_data(config.dataset)

#     train_vad(**config)



if __name__ == '__main__':

    a = torch.ones(1).to("cuda")  # to avoid race condition, use cuda device at the start
    print(a)
    # config = load_config()
    # main(config)
