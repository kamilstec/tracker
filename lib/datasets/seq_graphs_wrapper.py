from torch.utils.data import Dataset

from .seq_graphs import MOT17Graph, MOT17GraphTrain, MOT17GraphVal


class MOT17GraphWrapper(Dataset):

    def __init__(self, split, args):
        train_sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
        test_sequences = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']
        if "train" == split:
            sequences = train_sequences
        elif "test" == split:
            sequences = test_sequences
        elif "all" == split:
            sequences = train_sequences + test_sequences
        elif f"MOT17-{split}" in train_sequences + test_sequences:
            sequences = [f"MOT17-{split}"]

        self.data = []
        for s in sequences:
            self.data.append(MOT17Graph(seq_name=s, **args))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MOT17GraphTrainWrapper(Dataset):

    def __init__(self, split, args):
        train_sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
        test_sequences = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']
        if "train" == split:
            sequences = train_sequences
        elif "test" == split:
            sequences = test_sequences
        #elif 'train_small' == split:
        #    sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05']
        elif "all" == split:
            sequences = train_sequences + test_sequences
        elif f"MOT17-{split}" in train_sequences + test_sequences:
            sequences = [f"MOT17-{split}"]

        self.data = []
        for s in sequences:
            self.data.append(MOT17GraphTrain(seq_name=s, **args))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MOT17GraphValWrapper(Dataset):

    def __init__(self, split, args):
        train_sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
        test_sequences = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']
        if "train" == split:
            sequences = train_sequences
        elif "test" == split:
            sequences = test_sequences
        #elif 'train_small' == split:
        #    sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05']
        elif "all" == split:
            sequences = train_sequences + test_sequences
        elif f"MOT17-{split}" in train_sequences + test_sequences:
            sequences = [f"MOT17-{split}"]

        self.data = []
        for s in sequences:
            self.data.append(MOT17GraphVal(seq_name=s, **args))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
