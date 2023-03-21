import argparse
from pathlib import Path

from utils.general import increment_path
from utils.dataloaders import create_dataloader
import torch
import torch.nn as nn
import torch.optim as optim


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='DCOILBRENTEU.csv', help='data path')
    parser.add_argument('--epochs', type=int, default=20,
                        help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=24,
                        help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--workers', type=int, default=4,
                        help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--seq-len', type=int, default=12,
                        help='seq len')
    return parser.parse_known_args()[0]

def main(opt):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seq_len = opt.seq_len
    train_loader = create_dataloader(path=opt.dataset, batch_size=opt.batch_size, workers=opt.workers, seq_len=seq_len)
    
    for epoch in range(opt.epochs):  # loop over the dataset multiple times
        for input, output in train_loader:
            input = input.to(device, non_blocking=True)
            output = output.to(device, non_blocking=True)
            print('input  ',input.shape)
            print(output.shape,'\n')
            return

    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)