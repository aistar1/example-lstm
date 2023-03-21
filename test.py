
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from models.model import LSTM
from sklearn.preprocessing import MinMaxScaler



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='runs/train-cls/exp/last.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--workers', type=int, default=4,
                        help='max dataloader workers (per RANK in DDP mode)')
    return parser.parse_known_args()[0]

def main(opt):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LSTM()
    model.load_state_dict(torch.load(opt.weight))
    model.eval()
    model.to(device)


    test_data_size = 12
    flight_data = sns.load_dataset("flights")
    all_data = flight_data['passengers'].values.astype(float)
    train_data = all_data[:-test_data_size]
    test_data = all_data[-test_data_size:]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))
    fut_pred = 12
    train_window = 12

    test_inputs = train_data_normalized[-train_window:].tolist()




    seq = torch.FloatTensor(test_inputs[-train_window:])
    seq = seq.to(device)
    seq = seq.unsqueeze(1)

    for i in range(train_window):
        with torch.no_grad():
            test_inputs.append(model(seq[i]).item())

    actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))

    x = np.arange(132, 144, 1)
    plt.title('Month vs Passenger')
    plt.ylabel('Total Passengers')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(flight_data['passengers'])
    plt.plot(x,actual_predictions)
    plt.savefig('result.jpg')
    plt.close()



    plt.title('Month vs Passenger')
    plt.ylabel('Total Passengers')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(flight_data['passengers'][-train_window:])
    plt.plot(x,actual_predictions)
    plt.savefig('result2.jpg')
    plt.close()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
