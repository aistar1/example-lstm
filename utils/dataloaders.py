import torch
import os
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from utils.general import draw_pic

def create_dataloader(path,
                      batch_size,
                      workers,
                      seq_len):
    dataset = CustomImageDataset(path, seq_len)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
    return train_loader


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
        # print('train_seq  ', train_seq)
        # print('train_label  ', train_label)
        # print('\n  ')
    return inout_seq

class CustomImageDataset(Dataset):
    def __init__(self, path, seq_len):
        sns.get_dataset_names()
        flight_data = sns.load_dataset("flights")
        flight_data.head()
        draw_pic(flight_data)
        print(flight_data)
        print(flight_data.shape)
        '''
             year month  passengers
        0    1949   Jan         112
        1    1949   Feb         118
        2    1949   Mar         132
        3    1949   Apr         129
        4    1949   May         121
        ..    ...   ...         ...
        139  1960   Aug         606
        140  1960   Sep         508
        141  1960   Oct         461
        142  1960   Nov         390
        143  1960   Dec         432


        (144, 3)
        '''

        all_data = flight_data['passengers'].values.astype(float)
        test_data_size = 12
        train_data = all_data[:-test_data_size]
        test_data = all_data[-test_data_size:]

        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))
        train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
        self.train_inout_seq = create_inout_sequences(train_data_normalized, seq_len)

    def __len__(self):
        return len(self.train_inout_seq)

    def __getitem__(self, idx):
        train_seq = self.train_inout_seq[idx][0]
        train_label = self.train_inout_seq[idx][1]
        train_seq = train_seq.unsqueeze(1)

        return train_seq, train_label