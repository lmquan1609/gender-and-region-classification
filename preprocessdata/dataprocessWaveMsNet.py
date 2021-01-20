import torch
import pickle
from torch.utils.data import Dataset

def loaddata(filename):
    """Load data from pickle file

    Parameters
    ----------
    filename: str
        Path to file

    Returns
    -------
    data: list or dict
        Loaded file.

    """

    return pickle.load(open(filename, "rb"), encoding='latin1')


class WaveformDataset(Dataset):
    def __init__(self, pkl_file):

        self.sampleSet = loaddata(pkl_file)
        
    def __len__(self):
        return len(self.sampleSet)

    def __getitem__(self, index):

        data = self.sampleSet[index]['data']
        label = self.sampleSet[index]['label']

        data = torch.Tensor(data).unsqueeze(0)
        label = torch.LongTensor([label])

        return data, label

