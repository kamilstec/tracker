import torch
from torch_geometric.data import Data
from .detection import Detection


def build_graph(img, tracks, current_detections, distance_limit, test=True, mean_prediction=False):

    if len(tracks) and len(current_detections):
        node_attr = []
        edge_attr = []
        coords_original = []
        coords_normalized = []
        edges_first_row = []
        edges_second_row = []
        edges_complete_first_row = []
        edges_complete_second_row = []
        ground_truth = []

        for track in tracks:
            if mean_prediction == True:
                # Przewidywanie położenia
                temp = Detection(track.mean[:4], format='xyah')  # pobieramy przewidzine położenie z Kalmana w formacie xyah
                bbox = temp.to_tlbr()
                bbox_norm = bbox_normalization(img, temp.to_xywh())
            else:
                # Bez przewidywania położenia
                bbox = track.bbox
                bbox_norm = track.bbox_normalized

            coords_original.append(bbox) # oryginalne koordynaty w formacie tlbr do obliczania IoU
            coords_normalized.append(bbox_norm) # znormalizowane koordynaty w formacie xywh do porównywania różnicy w położeniu
            node_attr.append(track.crop)

        for detection in current_detections:
            coords_original.append(detection.bbox)
            coords_normalized.append(detection.bbox_normalized)
            node_attr.append(detection.crop)

        for i in range(len(tracks) + len(current_detections)):
            for j in range(len(tracks) + len(current_detections)):
                distance= ((coords_original[i][0]-coords_original[j][0])**2+(coords_original[i][1]-coords_original[j][1])**2)**0.5
                if i < len(tracks) and j >= len(tracks):
                    if distance<distance_limit:
                        edges_first_row.append(i)
                        edges_second_row.append(j)
                        edge_attr.append([0.0])
                    # tworzenie macierzy A
                    if test==True:
                        edges_complete_first_row.append(i)
                        edges_complete_second_row.append(j)
                    # tworzenie macierzy X_ref
                    if int(tracks[i].track_id) == int(current_detections[j-len(tracks)].track_id):
                        ground_truth.append(1.0)
                    else:
                        ground_truth.append(0.0)
                # połączenia nieskierowane
                elif i >= len(tracks) and j < len(tracks):
                    if distance<distance_limit:
                        edges_first_row.append(i)
                        edges_second_row.append(j)
                        edge_attr.append([0.0])

        frame_node_attr = torch.stack(node_attr)
        frame_edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        frame_edges_index = torch.tensor([edges_first_row, edges_second_row], dtype=torch.long)
        frame_coords_normalized = torch.tensor(coords_normalized, dtype=torch.float)
        frame_ground_truth = torch.tensor(ground_truth, dtype=torch.float)
        tracklets_frame = torch.tensor(len(tracks), dtype=torch.float).reshape(1)
        detections_frame = torch.tensor(len(current_detections), dtype=torch.float).reshape(1)
        coords_original = torch.tensor(coords_original, dtype= torch.float)
        edges_complete = torch.tensor([edges_complete_first_row, edges_complete_second_row], dtype=torch.long)

        data = Data(x=frame_node_attr, edge_index=frame_edges_index,
                    edge_attr=frame_edge_attr, coords_normalized=frame_coords_normalized,
                    coords_original=coords_original, ground_truth=frame_ground_truth,
                    det_num=detections_frame,
                    track_num=tracklets_frame, edges_complete=edges_complete)
        return data

def bbox_normalization(img, bbox):
    frame_height, frame_width, channels = img.shape
    bbox_norm = [bbox[0] / frame_width, bbox[1] / frame_height,
                 bbox[2] / frame_width, bbox[3] / frame_height]
    return bbox_norm
