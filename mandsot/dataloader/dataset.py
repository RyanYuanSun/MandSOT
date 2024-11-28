import torch
from torch.utils.data import Dataset


class VoiceDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        mfcc = torch.from_numpy(self.dataframe.iloc[idx]['mfcc']).float()
        onset = self.dataframe.iloc[idx]['onset']
        initial = self.dataframe.iloc[idx]['initials']
        return mfcc, onset
