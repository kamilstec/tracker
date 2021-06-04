import numpy as np
import torch
import cv2
from lib.nms import non_max_suppression
from lib.kalman_filter import KalmanFilter
from lib.datasets.other.detection import Detection
from lib.datasets.other.build_graph import build_graph
from lib.track import Track
#from lib.build_graph import build_graph
from lib.network.utils import hungarian
from torchvision.transforms import ToTensor
import torch_geometric  # tylko po to, żeby sprawdzić czy build_graph zwraca torch_geometric.data.data.Data
import matplotlib.pyplot as plt


class MyTracker(object):
    def __init__(self, min_similarity, min_confidence, use_kalman=False,
                 max_age=70, n_init=3, network=None):
        self.min_confidence = min_confidence
        self.nms_max_overlap = 1.0
        self.use_kalman = use_kalman
        self.kf = KalmanFilter()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network = network.to(self.device)

        self.min_similarity = min_similarity
        self.max_age = max_age
        self.graph_dist_limit = 200

        self.tracks = {}  # lista przechowująca ścieżki (obiekty klasy Track)
        self.next_id = 1  # id jakie otrzyma kolejna utworzona ścieżka
        self.results = {}
        self.kalman_results = {}
        self.kalman_results_for_lost = {}

    def reset(self):
        self.tracks = {}
        self.next_id = 1
        self.results = {}
        self.kalman_results = {}
        self.kalman_results_for_lost = {}

    def first_step(self, img, img_path, boxes, confidences):
        if boxes is None:
            return
        boxes = [boxes[i] for i, conf in enumerate(confidences) if conf > self.min_confidence]
        if not boxes:
            return

        crops = self.get_crops(img, img_path, boxes)
        for i, bbox in enumerate(boxes):
            detection = Detection(bbox, format='tlbr')
            bbox_norm = self.bbox_normalization(img, detection.to_xywh())
            mean, covariance = self.kf.initiate(detection.to_xyah())
            self.tracks[self.next_id] = Track(
                mean, covariance, bbox, bbox_norm, self.next_id, max_age=self.max_age, crop=crops[i]
            )
            self.next_id += 1

    def bbox_normalization(self, img, bbox):
        frame_height, frame_width, channels = img.shape
        bbox_norm = [bbox[0] / frame_width, bbox[1] / frame_height,
                     bbox[2] / frame_width, bbox[3] / frame_height]
        return bbox_norm

    def are_tracks_initialized(self):
        if self.next_id == 1:
            return False
        return True

    def predict(self):
        for track_id, track in self.tracks.items():
            track.predict(self.kf)

    def step(self, img, img_path, img_idx, boxes, confidences):
        """Predict Step"""
        self.predict()
        if boxes is None:
            return

        crops = self.get_crops(img, img_path, boxes)

        det_class_boxes = [Detection(box, format='tlbr') for box in boxes]
        detections = [Detection(det_class_boxes[i].to_tlbr(), self.bbox_normalization(img, det_class_boxes[i].to_xywh()),
                                conf, crops[i], -1)
                      for i, conf in enumerate(confidences) if conf > self.min_confidence]
        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracks = []
        for track_id, track in self.tracks.items():
            if not track.is_deleted():
                tracks.append(track)

        if not len(tracks) and len(detections):
            print('[!] Brak ścieżek w tej chwili')
            self.first_step(img, img_path, boxes, confidences)
            return
        if len(detections) and not len(tracks):
            print('[!] Brak detekcji w tej chwili')
            return
        if not len(tracks) and not len(detections):
            print('[!] Brak ścieżek i detekcji w tej chwili')
            self.first_step(img, img_path, boxes, confidences)
            return

        graph = build_graph(img, tracks, detections, distance_limit=self.graph_dist_limit, test=True,
                            mean_prediction=self.use_kalman)
        if not isinstance(graph, torch_geometric.data.Data) or len(graph.edge_index[0]) == 0:
            return

        graph.to(self.device)
        _, output, _, _, det_num, tracklet_num = self.network(graph)
        cost_matrix = output.view(tracklet_num, det_num)

        cleaned_output = hungarian(output, det_num, tracklet_num)
        cleaned_output = cleaned_output.view(tracklet_num, det_num)

        row_indices = []
        col_indices = []
        for i, row in enumerate(cleaned_output):
            for j, col in enumerate(row):
                if col == 1:
                    row_indices.append(i)
                    col_indices.append(j)

        matches, unmatched_tracks, unmatched_detections = [], [], []
        for col, detection in enumerate(detections):
            if col not in col_indices:
                unmatched_detections.append(detection)
        for row, track in enumerate(tracks):
            if row not in row_indices:
                unmatched_tracks.append(track)
        for row, col in zip(row_indices, col_indices):
            track = tracks[row]
            detection = detections[col]
            if cost_matrix[row, col] < self.min_similarity:
                unmatched_tracks.append(track)
                unmatched_detections.append(detection)
            else:
                matches.append((track, detection))

        for track, detection in matches:
            self.tracks[track.track_id].bbox = detection.bbox
            self.tracks[track.track_id].bbox_normalized = detection.bbox_normalized
            self.tracks[track.track_id].crop = detection.crop
            """Update Step"""
            self.tracks[track.track_id].update(self.kf, detection)
        for track in unmatched_tracks:
            self.tracks[track.track_id].mark_missed()
        for detection in unmatched_detections:
            # inicjacja nowej ścieżki
            mean, covariance = self.kf.initiate(detection.to_xyah())
            self.tracks[self.next_id] = Track(
                mean, covariance, detection.bbox, detection.bbox_normalized, self.next_id, max_age=self.max_age,
                crop=detection.crop
            )
            self.next_id += 1

        # Usuwamy ścieżki, które zostały skasowane w tej iteracji (dzięki funkcji mark_missed())
        to_delete = []
        for track_id, track in self.tracks.items():
            if track.is_deleted():
                to_delete.append(track_id)
        for track_id in to_delete:
            del self.tracks[track_id]

        ###  REZULTATY ###

        for track_id, track in self.tracks.items():
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            if track_id not in self.results.keys():
                self.results[track_id] = {}
            self.results[track_id][img_idx] = track.bbox

        for track_id, track in self.tracks.items():
            #if not track.is_confirmed():
            #    continue
            if track_id not in self.kalman_results.keys():
                self.kalman_results[track_id] = {}
            temp = Detection(track.mean[:4], format='xyah')
            predicted_bbox = temp.to_tlbr()
            self.kalman_results[track_id][img_idx] = predicted_bbox

        for track_id, track in self.tracks.items():
            if not track.is_lost():
                continue
            if track_id not in self.kalman_results_for_lost.keys():
                self.kalman_results_for_lost[track_id] = {}
            temp = Detection(track.mean[:4], format='xyah')
            predicted_bbox = temp.to_tlbr()
            self.kalman_results_for_lost[track_id][img_idx] = predicted_bbox

    def get_results(self):
        return self.results

    def get_kalman_results(self):
        return self.kalman_results

    def get_kalman_results_for_lost(self):
        return self.kalman_results_for_lost

    def get_crops(self, img, img_path, bbox_tlbr):
        """ramki w formacie tlbr"""
        crops = []
        for gt in bbox_tlbr:
            ori_img = plt.imread(img_path[0])
            height, width, channels = ori_img.shape

            gt[0] = np.clip(gt[0], 0, width - 1)
            gt[1] = np.clip(gt[1], 0, height - 1)
            gt[2] = np.clip(gt[2], 0, width - 1)
            gt[3] = np.clip(gt[3], 0, height - 1)
            img = ori_img[int(gt[1]):int(gt[3]), int(gt[0]):int(gt[2])]

            img = cv2.resize(img, (90, 150), interpolation=cv2.INTER_AREA)
            img = img / 255
            img = img.astype(np.float32)
            img -= [0.485, 0.456, 0.406]
            img /= [0.229, 0.224, 0.225]
            transform = ToTensor()
            img = transform(img)
            crops.append(img)
        return crops
