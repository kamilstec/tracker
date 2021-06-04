import configparser
import csv
import os
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from lib.datasets.config import cfg


class FakeDetector17:

    def __init__(self, seq_name=None, detector='FRCNN', format='tlwh'):
        """
        :param seq_name: Sekwencja na której przeprowadzamy śledzenie
        :param detector: Detektor użyty do wykonania detekcji (DPM/FRCNN/SDP)
        :param format: format w jakim zwracane mają być koordynaty ramek:
            tlwh: top left x, top left y, width, height - domyślny format MOTChallenge
            tlbr: top left x/y, bottom right x/y
            xyah: center x/y (koordynaty środka ramki), aspect ratio (stosunek szerokości do wysokości), wysokość
        """

        self._seq_name = seq_name
        self._faked_detector = detector
        self._format = format

        self._mot_det_dir = osp.join(cfg.DATA_DIR, 'MOT17Labels')
        self._test_folders = os.listdir(os.path.join(self._mot_det_dir, 'test'))
        self._train_folders = os.listdir(os.path.join(self._mot_det_dir, 'train'))

        self._unified_folders = {}
        for folder in self._test_folders:
            if folder[9:] == self._faked_detector:  # sprawdzamy czy końcówka to SDP lub DPM lub FRCNN
                self._unified_folders[folder[:8]] = osp.join('test', folder)  # mapujemy np. MOT17-02 na train/MOT17-02-FRCNN
        for folder in self._train_folders:
            if folder[9:] == self._faked_detector:
                self._unified_folders[folder[:8]] = osp.join('train', folder)

        if seq_name is not None:
            assert seq_name in self._unified_folders.keys(),\
                f'[!] Nie ma takiej sekwencji: {seq_name}'

        self.data = self.detections()

    def get_detections(self, idx):
        return self.data[idx-1]

    def detections(self):
        seq_path = osp.join(self._mot_det_dir, self._unified_folders[self._seq_name])

        config_file = osp.join(seq_path, 'seqinfo.ini')  # chcemy wydobyć informacje o długości sekwencji

        assert osp.exists(config_file), \
            'Brak pliku konfuguracyjnego: {}'.format(config_file)

        config = configparser.ConfigParser()
        config.read(config_file)
        seqLength = int(config['Sequence']['seqLength'])

        det_file = osp.join(seq_path, 'det', 'det.txt')

        dets = {}  # mapowanie nr_klatki: [detekcje_z_tej_klatki]

        for i in range(1, seqLength + 1):
            dets[i] = []

        if osp.exists(det_file):
            with open(det_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    if self._format == 'tlwh':
                        top_left_x = float(row[2]) - 1
                        top_left_y = float(row[3]) - 1
                        width = float(row[4])
                        height = float(row[5])
                        score = float(row[6])
                        bb = np.array([top_left_x, top_left_y, width, height, score], dtype=np.float32)
                    elif self._format == 'tlbr':
                        x1 = float(row[2]) - 1
                        y1 = float(row[3]) - 1
                        x2 = x1 + float(row[4])
                        y2 = y1 + float(row[5])
                        #score = float(row[6])
                        score = float(1)
                        bb = np.array([x1, y1, x2, y2, score], dtype=np.float32)
                    elif self._format == 'xyah':
                        center_x = float(row[2]) - 1 + float(row[4]) / 2
                        center_y = float(row[3]) - 1 + float(row[5]) / 2
                        height = float(row[5])
                        aspect_ratio = float(row[4]) / height
                        score = float(row[6])
                        bb = np.array([center_x, center_y, aspect_ratio, height, score], dtype=np.float32)
                    elif self._format == 'xywh':
                        center_x = float(row[2]) - 1 - float(row[4]) / 2
                        center_y = float(row[3]) - 1 - float(row[5]) / 2
                        height = float(row[5])
                        width = float(row[4])
                        score = float(row[6])
                        bb = np.array([center_x, center_y, width, height, score], dtype=np.float32)
                    dets[int(float(row[0]))].append(bb)

        dets_by_image = []  # detekcje poukładane według klatki do której należą
        for i in range(1, seqLength+1):
            #sample = {'img_id': i,
            #          'dets': dets[i]}
            sample = dets[i]
            dets_by_image.append(sample)

        return dets_by_image

class FakeDetector20:

    def __init__(self, seq_name=None, format='tlwh'):

        self._seq_name = seq_name
        self._format = format

        self._mot_det_dir = osp.join(cfg.DATA_DIR, 'MOT20')
        self._test_folders = os.listdir(os.path.join(self._mot_det_dir, 'test'))
        self._train_folders = os.listdir(os.path.join(self._mot_det_dir, 'train'))

        self._unified_folders = {}
        for folder in self._test_folders:
            self._unified_folders[folder[:8]] = osp.join('test', folder)
        for folder in self._train_folders:
            self._unified_folders[folder[:8]] = osp.join('train', folder)

        if seq_name is not None:
            assert seq_name in self._unified_folders.keys(),\
                f'[!] Nie ma takiej sekwencji: {seq_name}'

        self.data = self.detections()

    def get_detections(self, idx):
        return self.data[idx-1]

    def detections(self):
        seq_path = osp.join(self._mot_det_dir, self._unified_folders[self._seq_name])

        config_file = osp.join(seq_path, 'seqinfo.ini')

        assert osp.exists(config_file), \
            'Brak pliku konfiguracyjnego: {}'.format(config_file)

        config = configparser.ConfigParser()
        config.read(config_file)
        seqLength = int(config['Sequence']['seqLength'])

        det_file = osp.join(seq_path, 'det', 'det.txt')

        dets = {}  # mapowanie nr_klatki: [detekcje_z_tej_klatki]

        for i in range(1, seqLength + 1):
            dets[i] = []

        if osp.exists(det_file):
            with open(det_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    if self._format == 'tlwh':
                        top_left_x = float(row[2]) - 1
                        top_left_y = float(row[3]) - 1
                        width = float(row[4])
                        height = float(row[5])
                        score = float(row[6])
                        bb = np.array([top_left_x, top_left_y, width, height, score], dtype=np.float32)
                    elif self._format == 'tlbr':
                        x1 = float(row[2]) - 1
                        y1 = float(row[3]) - 1
                        x2 = x1 + float(row[4])
                        y2 = y1 + float(row[5])
                        #score = float(row[6])
                        score = float(1)
                        bb = np.array([x1, y1, x2, y2, score], dtype=np.float32)
                    elif self._format == 'xyah':
                        center_x = float(row[2]) - 1 + float(row[4]) / 2
                        center_y = float(row[3]) - 1 + float(row[5]) / 2
                        height = float(row[5])
                        aspect_ratio = float(row[4]) / height
                        score = float(row[6])
                        bb = np.array([center_x, center_y, aspect_ratio, height, score], dtype=np.float32)
                    elif self._format == 'xywh':
                        center_x = float(row[2]) - 1 - float(row[4]) / 2
                        center_y = float(row[3]) - 1 - float(row[5]) / 2
                        height = float(row[5])
                        width = float(row[4])
                        score = float(row[6])
                        bb = np.array([center_x, center_y, width, height, score], dtype=np.float32)
                    dets[int(float(row[0]))].append(bb)

        dets_by_image = []  # detekcje poukładane według klatki do której należą
        for i in range(1, seqLength+1):
            sample = dets[i]
            dets_by_image.append(sample)

        return dets_by_image


import re
class FakeDetector15:

    def __init__(self, seq_name=None, format='tlwh'):

        self._seq_name = seq_name
        self._format = format

        self._mot_det_dir = osp.join(cfg.DATA_DIR, '2DMOT2015')
        self._test_folders = os.listdir(os.path.join(self._mot_det_dir, 'test'))
        self._train_folders = os.listdir(os.path.join(self._mot_det_dir, 'train'))

        self._unified_folders = {}
        for folder in self._test_folders:
            self._unified_folders[folder] = osp.join('test', folder)  # mapujemy np. MOT17-02 na train/MOT17-02-FRCNN
        for folder in self._train_folders:
            self._unified_folders[folder] = osp.join('train', folder)

        if seq_name is not None:
            assert seq_name in self._unified_folders.keys(),\
                f'[!] Nie ma takiej sekwencji: {seq_name}'

        self.data = self.detections()

    def get_detections(self, idx):
        return self.data[idx-1]

    def detections(self):
        seq_path = osp.join(self._mot_det_dir, self._unified_folders[self._seq_name])

        im_dir = osp.join(seq_path, 'img1')
        det_file = osp.join(seq_path, 'det', 'det.txt')

        dets = {}  # mapowanie nr_klatki: [detekcje_z_tej_klatki]

        valid_files = [f for f in os.listdir(im_dir) if len(re.findall("^[0-9]{6}[.][j][p][g]$", f)) == 1]
        seqLength = len(valid_files)

        for i in range(1, seqLength + 1):
            dets[i] = []

        if osp.exists(det_file):
            with open(det_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    if len(row) > 0:
                        if self._format == 'tlwh':
                            top_left_x = float(row[2]) - 1
                            top_left_y = float(row[3]) - 1
                            width = float(row[4])
                            height = float(row[5])
                            score = float(row[6])
                            bb = np.array([top_left_x, top_left_y, width, height, score], dtype=np.float32)
                        elif self._format == 'tlbr':
                            x1 = float(row[2]) - 1
                            y1 = float(row[3]) - 1
                            x2 = x1 + float(row[4])
                            y2 = y1 + float(row[5])
                            #score = float(row[6])
                            score = float(1)
                            bb = np.array([x1, y1, x2, y2, score], dtype=np.float32)
                        elif self._format == 'xyah':
                            center_x = float(row[2]) - 1 + float(row[4]) / 2
                            center_y = float(row[3]) - 1 + float(row[5]) / 2
                            height = float(row[5])
                            aspect_ratio = float(row[4]) / height
                            score = float(row[6])
                            bb = np.array([center_x, center_y, aspect_ratio, height, score], dtype=np.float32)
                        elif self._format == 'xywh':
                            center_x = float(row[2]) - 1 - float(row[4]) / 2
                            center_y = float(row[3]) - 1 - float(row[5]) / 2
                            height = float(row[5])
                            width = float(row[4])
                            score = float(row[6])
                            bb = np.array([center_x, center_y, width, height, score], dtype=np.float32)
                        dets[int(float(row[0]))].append(bb)

        dets_by_image = []  # detekcje poukładane według klatki do której należą
        for i in range(1, seqLength+1):
            sample = dets[i]
            dets_by_image.append(sample)

        return dets_by_image
