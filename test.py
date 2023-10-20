import torch

if __name__ == '__main__':

    a = torch.ones(1).to("cuda")  
    print(a)
