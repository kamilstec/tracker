import cv2
from PIL import Image
import numpy as np
import os
import os.path as osp
from random import randint
from math import fabs

import torch
from torchvision.transforms import CenterCrop, Normalize, Compose, RandomHorizontalFlip, RandomCrop, ToTensor, \
    RandomResizedCrop
from torch_geometric.data import Data # tylko po to, żeby sprawdzić czy typ się zgadza

from .sequence import MOT17Sequence
from .config import get_data_dir
from .other.build_graph import build_graph
from .other.detection import Detection


class MOT17Graph(MOT17Sequence):
    """Dataset do trenowania sieci na grafach.
    """
    def __init__(self, seq_name, vis_threshold, crop_H, crop_W, transform,
                 normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225],
                 jump=5, frames_look_back=1, max_distance=250):
        super().__init__(seq_name=seq_name, vis_threshold=vis_threshold)

        self.crop_H = crop_H
        self.crop_W = crop_W

        if self.seq_name:
            self.output_dir = get_data_dir(osp.join('MOT17graph', self.seq_name))

        if transform == "random":
            self.transform = Compose([RandomCrop((crop_H, crop_W)), RandomHorizontalFlip(), ToTensor(),
                                      Normalize(normalize_mean, normalize_std)])
        elif transform == "center":
            self.transform = Compose([CenterCrop((crop_H, crop_W)), ToTensor(), Normalize(normalize_mean,
                                                                                          normalize_std)])
        else:
            raise NotImplementedError(f'[!] Nie ma takiej transformacji: {transform}')

        self.jump = jump
        self.frames_look_back = frames_look_back
        self.max_distance = max_distance
        self.build_samples()

    def __getitem__(self, idx):
        detections_data, tracks_data = self.data[idx]
        detections = []
        tracks = []
        for detection_data in detections_data:
            crop = cv2.imread(detection_data['crop_path'])
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = Image.fromarray(crop)
            crop = self.transform(crop)
            detections.append(
                Detection(bbox=detection_data['person_bbox'],
                          bbox_normalized=detection_data['bbox_normalized'],
                          crop=crop, track_id=detection_data['person_id'], format='xywh'))
        for track_data in tracks_data:
            crop = cv2.imread(track_data['crop_path'])
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = Image.fromarray(crop)
            crop = self.transform(crop)
            tracks.append(
                Detection(bbox=track_data['person_bbox'],
                          bbox_normalized=track_data['bbox_normalized'],
                          crop=crop, track_id=track_data['person_id'], format='xywh'))
        graph = build_graph(None, tracks, detections, self.max_distance, test=False)

        return graph

    def build_samples(self):
        graphs = []

        frame0 = self.data[0]
        img = cv2.imread(frame0['img_path'])
        h, w, c = img.shape

        for frame_idx in range(1, len(self.data), self.jump):
            frame = self.data[frame_idx]
            frame_path = frame['img_path']
            frame_gt = frame['gt']

            detections = []
            detections_temp = [] # do testowania czy graf może powstać
            tracks = []
            tracks_temp = []  # to testowania czy graf może powstać
            for person_id, person_bbox in frame_gt.items(): # person_bbox w formacie tlbr
                data = {}
                data['person_id'] = person_id
                person_bbox_ = Detection(person_bbox, format='tlbr')
                data['person_bbox'] = person_bbox_.to_tlbr() # do obliczania IoU, pozostaje tlbr
                data['bbox_normalized'] = self.bbox_normalization(h, w, c, person_bbox_.to_xywh()) # do obliczania różnicy w położeniu; zamieniamy z tlbr na xywh
                crop_path = osp.join(self.output_dir, str(person_id) +
                                     '_' + str(frame_idx+1) + '_detekcja_' +
                                     '_' + frame_path[-10:-4] + '.png')
                if not os.path.exists(crop_path):  # Sprawdzamy czy ramka już istnieje
                    cv2.imwrite(crop_path, self.build_crop(frame_path, person_bbox_.to_tlbr()))
                data['crop_path'] = crop_path
                detections.append(data)

                crop = cv2.imread(crop_path)
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = Image.fromarray(crop)
                transform = Compose([ToTensor()])
                crop = transform(crop)
                detections_temp.append(
                    Detection(bbox=person_bbox_.to_tlbr(), bbox_normalized=self.bbox_normalization(h, w, c, person_bbox_.to_xywh()),
                              crop=crop, track_id=person_id, format='tlbr'))

                match = False
                prev_frame_idx = randint(max(0, frame_idx-self.frames_look_back), frame_idx-1)
                prev_frame = self.data[prev_frame_idx]
                prev_frame_gt = prev_frame['gt']
                if person_id in prev_frame_gt.keys():
                    person_bbox = prev_frame_gt[person_id]
                    match = True
                if not match:
                    continue
                prev_frame_path = prev_frame['img_path']
                data = {}
                data['person_id'] = person_id
                person_bbox_ = Detection(person_bbox, format='tlbr')
                data['person_bbox'] = person_bbox_.to_tlbr()  # zamieniamy z tlbr na xywh
                data['bbox_normalized'] = self.bbox_normalization(h, w, c, person_bbox_.to_xywh())
                crop_path = osp.join(self.output_dir, str(person_id) +
                                     '_' + str(frame_idx+1) + '_sciezka_' +
                                     '_' + prev_frame_path[-10:-4] + '.png')
                if not os.path.exists(crop_path):
                    cv2.imwrite(crop_path, self.build_crop(prev_frame_path, person_bbox_.to_tlbr()))
                data['crop_path'] = crop_path
                tracks.append(data)

                crop = cv2.imread(crop_path)
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = Image.fromarray(crop)
                transform = Compose([ToTensor()])
                crop = transform(crop)
                tracks_temp.append(
                    Detection(bbox=person_bbox_.to_tlbr(), bbox_normalized=self.bbox_normalization(h, w, c, person_bbox_.to_xywh()),
                              crop=crop, track_id=person_id, format='tlbr'))

            graph = build_graph(None, tracks_temp, detections_temp, self.max_distance, test=False)
            if isinstance(graph, Data):
                if len(graph.edge_index[0]) != 0:
                    graphs.append((detections, tracks))
                else:
                    print(f'(0) Nie udało się utworzyć grafu dla klatki {frame_idx} z sekwencji {self.seq_name}')
            if not isinstance(graph, Data):
                print(f'(None) Nie udało się utworzyć grafu dla klatki {frame_idx} z sekwencji {self.seq_name}')
        self.data = graphs # dane do zbudowania grafu

    def build_crop(self, img_path, gt):
        img = cv2.imread(img_path)

        height, width, channels = img.shape
        gt[0] = np.clip(gt[0], 0, width - 1)
        gt[1] = np.clip(gt[1], 0, height - 1)
        gt[2] = np.clip(gt[2], 0, width - 1)
        gt[3] = np.clip(gt[3], 0, height - 1)
        img = img[int(gt[1]):int(gt[3]), int(gt[0]):int(gt[2])]

        img = cv2.resize(img, (int(self.crop_W * 1.125), int(self.crop_H * 1.125)), interpolation=cv2.INTER_LINEAR)

        return img

    def bbox_normalization(self, frame_height, frame_width, channels, bbox):
        bbox_norm = [bbox[0] / frame_width, bbox[1] / frame_height,
                     bbox[2] / frame_width, bbox[3] / frame_height]
        return bbox_norm


class MOT17GraphTrain(MOT17Graph):
    def __init__(self, seq_name, vis_threshold, crop_H, crop_W, transform,
                 normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225],
                 jump=5, frames_look_back=1, max_distance=250):
        super().__init__(seq_name=seq_name, vis_threshold=vis_threshold, crop_H=crop_H, crop_W=crop_W,
                         transform=transform, normalize_mean=normalize_mean, normalize_std=normalize_std,
                         jump=jump, frames_look_back=frames_look_back, max_distance=max_distance)
        print(f'Skończono inicjować grafy dla zbioru treningowego: {self.seq_name}')
        len_data = int(len(self.data) * 0.8)
        self.data = self.data[:len_data]


class MOT17GraphVal(MOT17Graph):
    def __init__(self, seq_name, vis_threshold, crop_H, crop_W, transform,
                 normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225],
                 jump=5, frames_look_back=1, max_distance=250):
        super().__init__(seq_name=seq_name, vis_threshold=vis_threshold, crop_H=crop_H, crop_W=crop_W,
                         transform=transform, normalize_mean=normalize_mean, normalize_std=normalize_std,
                         jump=jump, frames_look_back=frames_look_back, max_distance=max_distance)
        print(f'Skończono inicjować grafy dla zbioru walidacyjnego: {self.seq_name}')
        len_data = int(len(self.data) * 0.8)
        self.data = self.data[len_data:]
