from torch.utils.data import Dataset

from .sequence import MOT17Sequence, MOT20Sequence, MOT15Sequence


class MOT17Wrapper(Dataset):

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
        else:
            raise NotImplementedError(f'Brak takiej sekwencji: {split}')

        self._data = []
        for s in sequences:
            self._data.append(MOT17Sequence(seq_name=s, **args))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class MOT20Wrapper(MOT17Wrapper):

    def __init__(self, split, args):
        train_sequences = ['MOT20-01', 'MOT20-02', 'MOT20-03']#, 'MOT20-05']
        test_sequences = ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08']

        if 'train' == split:
            sequences = train_sequences
        elif 'test' == split:
            sequences = test_sequences
        elif 'all' == split:
            sequences = train_sequences + test_sequences
        elif f'MOT20-{split}' in train_sequences + test_sequences:
            sequences = [f'MOT20-{split}']
        else:
            raise NotImplementedError(f'Brak takiej sekwencji: {split}')

        self._data = []
        for s in sequences:
            self._data.append(MOT20Sequence(seq_name=s, **args))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class MOT15Wrapper(Dataset):

    def __init__(self, split, dataloader):

        #train_sequences = ['Venice-2', 'KITTI-17', 'KITTI-13', 'ADL-Rundle-8', 'ADL-Rundle-6', 'ETH-Pedcross2',
        #                   'ETH-Sunnyday', 'ETH-Bahnhof', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte']
        train_sequences = ['KITTI-17', 'KITTI-13', 'ETH-Sunnyday', 'ETH-Bahnhof', 'PETS09-S2L1', 'TUD-Campus',
                           'TUD-Stadtmitte']
        test_sequences = ['Venice-1', 'KITTI-19', 'KITTI-16', 'ADL-Rundle-3', 'ADL-Rundle-1', 'AVG-TownCentre',
                          'ETH-Crossing', 'ETH-Linthescher', 'ETH-Jelmoli', 'PETS09-S2L2', 'TUD-Crossing']

        if "train" == split:
            sequences = train_sequences
        elif "test" == split:
            sequences = test_sequences
        elif "train_static" == split:
            sequences = ['KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte']
        elif "train_dynamic" == split:
            sequences = ['KITTI-13', 'ETH-Sunnyday', 'ETH-Bahnhof']
        elif split in train_sequences or split in test_sequences:
            sequences = [split]
        else:
            raise NotImplementedError(f'Brak takiej sekwencji: {split}')

        self._data = []

        for s in sequences:
            self._data.append(MOT15Sequence(seq_name=s, **dataloader))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
