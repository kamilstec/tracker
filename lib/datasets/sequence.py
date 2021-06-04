import configparser
import csv
import os
import os.path as osp

import numpy as np
from PIL import Image
import cv2
from torchvision import transforms as T
from torch.utils.data import Dataset

from .config import cfg


class MOT17Sequence(Dataset):
    def __init__(self, seq_name=None, vis_threshold=0.0,
                 normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        self.seq_name = seq_name
        self.vis_threshold = vis_threshold

        self.mot_dir = osp.join(cfg.DATA_DIR, 'MOT17Det')
        self.train_folders = os.listdir(os.path.join(self.mot_dir, 'train'))
        self.test_folders = os.listdir(os.path.join(self.mot_dir, 'test'))

        self.transforms = T.Compose([T.ToTensor(), T.Normalize(normalize_mean,
                                                               normalize_std)])

        if seq_name:
            assert seq_name in self.train_folders or seq_name in self.test_folders, \
                f'[!] Nie ma takiej sekwencji: {seq_name}'

            self.data, self.no_gt = self.sequence()
        else:
            self.data = []
            self.no_gt = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        img = Image.open(data['img_path']).convert("RGB")
        img = self.transforms(img)

        sample = {}
        sample['img'] = img
        sample['img_path'] = data['img_path']
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']

        return sample

    def sequence(self):
        seq_name = self.seq_name
        if seq_name in self.train_folders:
            seq_path = osp.join(self.mot_dir, 'train', seq_name)
        else:
            seq_path = osp.join(self.mot_dir, 'test', seq_name)

        config_file = osp.join(seq_path, 'seqinfo.ini')

        assert osp.exists(config_file), \
            f'Plik konfiguracyjny nie istnieje: {config_file}'

        config = configparser.ConfigParser()
        config.read(config_file)
        seqLength = int(config['Sequence']['seqLength'])

        imDir = osp.join(seq_path, 'img1')
        gt_file = osp.join(seq_path, 'gt', 'gt.txt')

        visibility = {}
        boxes = {}

        for i in range(1, seqLength + 1):
            visibility[i] = {}
            boxes[i] = {}

        no_gt = False
        if osp.exists(gt_file):
            with open(gt_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= self.vis_threshold:
                        x1 = int(row[2]) - 1 # x_tl bez zmian
                        y1 = int(row[3]) - 1 # y_tl bez zmian
                        x2 = x1 + int(row[4]) - 1 # w => x_br
                        y2 = y1 + int(row[5]) - 1 # h => y_br
                        bb = np.array([x1, y1, x2, y2], dtype=np.float32)
                        boxes[int(row[0])][int(row[1])] = bb
                        visibility[int(row[0])][int(row[1])] = float(row[8])
        else:
            no_gt = True

        samples = []
        for i in range(1, seqLength + 1):
            img_path = osp.join(imDir, "{:06d}.jpg".format(i))
            sample = {'img_path': img_path,
                      'gt': boxes[i],
                      'vis': visibility[i],}

            samples.append(sample)

        return samples, no_gt


class MOT20Sequence(MOT17Sequence):

    def __init__(self, seq_name=None, vis_threshold=0.0,
                 normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        self._seq_name = seq_name
        self._vis_threshold = vis_threshold

        self._mot_dir = osp.join(cfg.DATA_DIR, 'MOT20')
        self._train_folders = os.listdir(os.path.join(self._mot_dir, 'train'))
        self._test_folders = os.listdir(os.path.join(self._mot_dir, 'test'))

        self.transforms = T.Compose([T.ToTensor(), T.Normalize(normalize_mean,
                                                               normalize_std)])

        if seq_name:
            assert seq_name in self._train_folders or seq_name in self._test_folders, \
                f'[!] Nie ma takiej sekwencji: {seq_name}'

            self.data, self.no_gt = self.sequence()

            # UWAGA: ponieważ sekwencje MOT20 są mega długie, wycinam z nich tylko mały fragment (pierwsze 350
            # klatek, co przekłada się na pierwsze 14 sekund każdej sekwencji (mają 25fps))
            self.data = self.data[:350]#[:350]
        else:
            self.data = []
            self.no_gt = True


import re
class MOT15Sequence(MOT17Sequence):
    def __init__(self, seq_name=None, vis_threshold=0.0, normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225]):
        self.seq_name = seq_name
        self.vis_threshold = vis_threshold

        self.mot_dir = osp.join(cfg.DATA_DIR, '2DMOT2015')

        self.train_folders = os.listdir(os.path.join(self.mot_dir, 'train'))
        self.test_folders = os.listdir(os.path.join(self.mot_dir, 'test'))

        self.transforms = T.Compose([T.ToTensor(), T.Normalize(normalize_mean,
                                                               normalize_std)])

        if seq_name:
            assert seq_name in self.train_folders or seq_name in self.test_folders, \
                f'[!] Nie ma takiej sekwencji: {seq_name}'

            self.data, self.no_gt = self.sequence()
        else:
            self.data = []
            self.no_gt = True

    def sequence(self):
        seq_name = self.seq_name
        if seq_name in self.train_folders:
            seq_path = osp.join(self.mot_dir, 'train', seq_name)
        else:
            seq_path = osp.join(self.mot_dir, 'test', seq_name)

        imDir = osp.join(seq_path, 'img1')
        gt_file = osp.join(seq_path, 'gt', 'gt.txt')

        total = []

        boxes = {}
        visibility = {}

        valid_files = [f for f in os.listdir(imDir) if len(re.findall("^[0-9]{6}[.][j][p][g]$", f)) == 1]
        seqLength = len(valid_files)

        for i in range(1, seqLength+1):
            boxes[i] = {}
            visibility[i] = {}

        no_gt = False
        if osp.exists(gt_file):
            with open(gt_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    if int(row[6]) == 1:
                        x1 = int(row[2]) - 1
                        y1 = int(row[3]) - 1
                        x2 = x1 + float(row[4]) - 1
                        y2 = y1 + float(row[5]) - 1
                        bb = np.array([x1,y1,x2,y2], dtype=np.float32)
                        boxes[int(row[0])][int(row[1])] = bb
                        visibility[int(row[0])][int(row[1])] = float(row[8])
        else:
            no_gt = True

        for i in range(1,seqLength+1):
            im_path = osp.join(imDir,"{:06d}.jpg".format(i))

            sample = { 'gt': boxes[i],
                       'img_path': im_path,
                       'vis': visibility[i],
            }

            total.append(sample)

        return total, no_gt
