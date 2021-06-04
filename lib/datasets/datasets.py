from .sequence_wrapper import MOT17Wrapper, MOT20Wrapper, MOT15Wrapper
from .seq_graphs_wrapper import MOT17GraphWrapper, MOT17GraphTrainWrapper, MOT17GraphValWrapper


_sets = {}

for split in ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08',
              '09', '10', '11', '12', '13', '14']:
    name = f'mot17_{split}'
    _sets[name] = (lambda *args, split=split: MOT17Wrapper(split, *args))

for split in ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08']:
    name = f'mot20_{split}'
    _sets[name] = (lambda *args, split=split: MOT20Wrapper(split, *args))

for split in ['train', 'test', 'train_static', 'train_dynamic', 'Venice-2', 'KITTI-17', 'KITTI-13', 'ADL-Rundle-8',
              'ADL-Rundle-6', 'ETH-Pedcross2', 'ETH-Sunnyday', 'ETH-Bahnhof', 'PETS09-S2L1', 'TUD-Campus',
              'TUD-Stadtmitte']:
    name = f'mot15_{split}'
    _sets[name] = (lambda *args, split=split: MOT15Wrapper(split, *args))

for split in ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08',
              '09', '10', '11', '12', '13', '14']:
    name = f'mot17graph_{split}'
    _sets[name] = (lambda *args, split=split: MOT17GraphWrapper(split, *args))

for split in ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08',
              '09', '10', '11', '12', '13', '14']:
    name = f'mot17graph_train_{split}'
    _sets[name] = (lambda *args, split=split: MOT17GraphTrainWrapper(split, *args))

for split in ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08',
              '09', '10', '11', '12', '13', '14']:
    name = f'mot17graph_val_{split}'
    _sets[name] = (lambda *args, split=split: MOT17GraphValWrapper(split, *args))


class Datasets:

    def __init__(self, dataset, *args):
        assert dataset in _sets, f'[!] Nie znaleziono zbioru: {dataset}'

        if len(args) == 0:
            args = [{}]

        self._data = _sets[dataset](*args)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
