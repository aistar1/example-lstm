import argparse
from pathlib import Path

from utils.general import increment_path
from utils.dataloaders import create_dataloader
from models.model import LSTM
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='649bigNew.txt', help='data path')
    parser.add_argument('--epochs', type=int, default=150,
                        help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--workers', type=int, default=4,
                        help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--seq-len', type=int, default=4,
                        help='seq len')
    return parser.parse_known_args()[0]

def main(opt):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Parameters
    save_dir = increment_path(
        Path('runs/train-cls') / 'exp', exist_ok=None)  # increment run

    # Directories
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = save_dir / 'last.pt', save_dir / 'best.pt'

    seq_len = opt.seq_len
    train_loader = create_dataloader(
        path=opt.dataset, batch_size=opt.batch_size, workers=opt.workers, seq_len=seq_len)

    model = LSTM()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(opt.epochs):  # loop over the dataset multiple times
        tloss = 0.0  # train loss
        # model.train()

        for input, target in train_loader:

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            #output_ = output.unsqueeze(1)
            #output_ = output

            optimizer.zero_grad()
            output_pred = model(input.float())

            # print('input  ',input)
            # print('output_pred  ',output_pred.shape)
            # print(output.shape,'\n')

            loss = criterion(output_pred, target.float())
            loss.backward()

            optimizer.step()
            tloss += loss.item() * input.size(0)

        print(f"Epoch {epoch+1}: Loss = {tloss / len(train_loader):.4f}")
        
        if (epoch) % 10 == 0:
            torch.save(model.state_dict(), last)

    torch.save(model.state_dict(), last)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
