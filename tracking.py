import os
import time
from os import path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader

import motmetrics as mm
mm.lap.default_solver = 'lap'

from tqdm import tqdm
from sacred import Experiment
from sacred.observers import FileStorageObserver
from lib.datasets.config import get_output_dir, cfg
from lib.datasets.datasets import Datasets

from lib.utils import plot_sequence, plot_sequence_ids_frames, plot_sequence_ids_frames_kalman, get_mot_accum, evaluate_mot_accums
from lib.detector.fake_detector import FakeDetector17, FakeDetector20, FakeDetector15
from lib.my_tracker import MyTracker
from lib.network.complete_net import completeNet

ex = Experiment('Tracking')

ex.observers.append(FileStorageObserver('../gdrive/MyDrive/KamMOT_trenowanie/logs'))

ex.add_config('config/tracking.yaml')

@ex.automain
def my_main(_config, _log, _run, tracking):

    #torch.manual_seed(tracking['seed'])
    #torch.cuda.manual_seed(tracking['seed'])
    #np.random.seed(tracking['seed'])
    #torch.backends.cudnn.deterministic = True

    print(_config)

    output_dir = osp.join(get_output_dir(tracking['name']) + '_runID_' + _run._id)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    _log.info("Initializing everything.")

    # detektory
    obj_detect_all = {}
    obj_detect17 = {}
    obj_detect20 = {}
    obj_detect15 = {}
    obj_detect_all['mot17'] = obj_detect17
    obj_detect_all['mot20'] = obj_detect20
    obj_detect_all['mot15'] = obj_detect15
    for seq in ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']:
        pass
        #obj_detect17[seq] = FakeDetector17(seq_name=seq, detector='FRCNN', format='tlbr')
    for seq in ['MOT20-01', 'MOT20-02', 'MOT20-03']:#, 'MOT20-05']:
        pass
        #obj_detect20[seq] = FakeDetector20(seq_name=seq, format='tlbr')#'xywh')
    for seq in ['Venice-2', 'KITTI-17', 'KITTI-13', 'ADL-Rundle-8', 'ADL-Rundle-6', 'ETH-Pedcross2',
                'ETH-Sunnyday', 'ETH-Bahnhof', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte']:
        obj_detect15[seq] = FakeDetector15(seq_name=seq, format='tlbr')
    if tracking['dataset'][:5] == 'mot17':
        obj_detect = obj_detect_all['mot17']
    elif tracking['dataset'][:5] == 'mot20':
        obj_detect = obj_detect_all['mot20']
    elif tracking['dataset'][:5] == 'mot15':
        obj_detect = obj_detect_all['mot15']
    else:
        print(f"Nieznany dataset: {tracking['dataset']}")

    if not torch.cuda.is_available():
        _log.warning('Obliczenia wykonywane na CPU. Będzie wolno.')

    # tracker
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn_encoder = str(torch.load(tracking['model_path'], map_location=device)['cnn_encoder'])
    #print(f'Użyty encoderCNN: {cnn_encoder}')
    model = completeNet(cnn_encoder)
    model.load_state_dict(torch.load(tracking['model_path'], map_location=device)['model_state_dict']) # colab

    model = model.to(device)
    model.eval()
    tracker = MyTracker(**tracking['tracker'], network=model)

    # dataset
    dataset = Datasets(tracking['dataset'])

    time_total = 0  # czas potrzebny na przerobienie wszystkich sekwencji
    num_frames = 0  # ilość przerobionych klatek ze wszystkich sekwencji
    mot_accums = []  # tutaj przechowujemy statystyki dotyczące śledzenia dla każdej sekwencji z osobna

    for seq_idx, seq in enumerate(dataset):

        obj_det = obj_detect[seq.seq_name]

        start = time.time()

        _log.info(f"Tracking: {seq.seq_name}")

        tracker.reset()

        data_loader = DataLoader(seq, batch_size=1, shuffle=False)
        for frame_idx, frame in enumerate(tqdm(data_loader), 0):

            sample = obj_det.get_detections(frame_idx+1)
            sample = np.asarray(sample)
            if sample.size != 0: # istnieją jakieś detekcje dla tej klatki
                dets = sample[:, :4]
                scores = sample[:, 4]
            else: # nie było detekcji dla tej klatki
                dets = None
                scores = None
            if len(seq) * tracking['frame_split'][0] <= frame_idx <= len(seq) * tracking['frame_split'][1]:
                with torch.no_grad():
                    frame_path = frame['img_path']
                    frame = frame['img']

                    frame = frame.numpy()
                    frame = np.swapaxes(frame, 1, 3)
                    frame = frame[0]
                    frame = np.swapaxes(frame, 0, 1)  # (h, w, 3)

                    if not tracker.are_tracks_initialized():
                        tracker.first_step(frame, frame_path, dets, scores)
                    else:
                        #tracker.predict()
                        tracker.step(frame, frame_path, frame_idx+1, dets, scores)
                num_frames += 1
            else:
                break

        time_total += time.time() - start

        results = tracker.get_results()
        kalman_results = tracker.get_kalman_results()
        kalman_results_for_lost = tracker.get_kalman_results_for_lost()


        _log.info(f"Tracks found: {len(results)}")
        #_log.info(f"Tracks found: {len(kalman_results)}")
        _log.info(f"Runtime for {seq.seq_name}: {time.time() - start :.2f} s.")

        if seq.no_gt:
            _log.info(f"Nie ma danych GT.")
        else:
            mot_accums.append(get_mot_accum(results, seq))

        if tracking['write_images']:
            _log.info(f"Plotting whole sequence with tracking results to: {output_dir}.")

            plot_sequence(results, seq, osp.join(output_dir, tracking['dataset'], str(seq.seq_name)))
            plot_sequence(kalman_results, seq, osp.join(output_dir, tracking['dataset'], str(seq.seq_name)+'_kalman'))
            plot_sequence(kalman_results_for_lost, seq, osp.join(output_dir, tracking['dataset'], str(seq.seq_name)+'_kalman_lost'))

            #ids_frames = [(4, 30),(4, 31)]
            #plot_sequence_ids_frames_kalman(ids_frames, results, kalman_results, seq, osp.join(output_dir, tracking['dataset'], str(seq.seq_name)+'_resnet50_1_k_dk_1')) # kalman, detekcje i kalman, 1-porówanie z i bez kalmana
            #ids_frames = [(4, 36),(4, 37),(4, 38),(4, 39),(4, 40)]
            #ids_frames = [(1, 26), (1, 39)]
            #plot_sequence_ids_frames(ids_frames, kalman_results, True, seq, osp.join(output_dir, tracking['dataset'], str(seq.seq_name)+'_resnet50_1_k_k_1')) # kalman, sam kalman
            """
            ### Odkomentować dla Resnet'a50 trenowanego na F_back=1
            ## Porównanie ze sobą resnet'a50 trenowanego na F_back=1 z i bez Kalmana
            # wyświetlenie ramek z detektora i przewidzianych przez Kalmana. ograniczone do wybranych ścieżek i klatek (ścieżka,klatka)
            ids_frames = [(6, 27), (6, 32), (6, 38), (6, 48)] # <- resnet50/1 z Kalmanem
            plot_sequence_ids_frames_kalman(ids_frames, results, kalman_results, seq, osp.join(output_dir, tracking['dataset'], str(seq.seq_name)+'_resnet50_1_k_dk_1')) # kalman, detekcje i kalman, 1-porówanie z i bez kalmana
            # wyświetlenie co robi kalman pomiędzy klatkami
            ids_frames = [(6, 27),(6, 28),(6, 29),(6, 30),(6, 31),(6, 32), # <- resnet50/1 z Kalmanem
                          (6,38),(6, 39),(6, 40),(6, 41),(6, 42),(6, 43),(6, 44),(6, 45),(6, 46),(6, 47),(6, 48)]
            plot_sequence_ids_frames(ids_frames, kalman_results, True, seq, osp.join(output_dir, tracking['dataset'], str(seq.seq_name)+'_resnet50_1_k_k_1')) # kalman, sam kalman
            # wyświetlenie ramek z detektora
            ids_frames = [(6, 27), (2, 32), (2, 38), (5, 48),(7, 57),(13, 77)] # <- resnet50/1 bez Kalmana
            plot_sequence_ids_frames(ids_frames, results, False, seq, osp.join(output_dir, tracking['dataset'], str(seq.seq_name) + '_resnet50_1_bk_d_1')) # bez kalmana, same detekcje
            ## Porównanie ze sobą resnet'a50 trenowanego na F_back=1 z Kalmanem i resnet'a50 trenowanego na F_back=30 bez Kalmana
            # wyświetlenie ramek z detektora i przewidzianych przez Kalmana
            ids_frames = [(19, 57), (19, 77)] # <- resnet50/1 z Kalmanem
            plot_sequence_ids_frames(ids_frames, results, False, seq, osp.join(output_dir, tracking['dataset'], str(seq.seq_name) + '_resnet50_1_k_d_2'))# 2- drugi test
            """

            """
            ### Odkomentować dla Resnet'a50 trenowanego na F_back=30
            ids_frames = [(6, 27), (1, 32), (1, 38), (12, 48),(4, 57), (4, 77)]  # <- resnet50/30 bez Kalmana
            plot_sequence_ids_frames(ids_frames, results, False, seq, osp.join(output_dir, tracking['dataset'], str(seq.seq_name) + '_resnet50_30_bk_d_2'))  # 2- drugi test
            """

    _log.info(f"Tracking runtime for all sequences (without evaluation or image writing): "
              f"{time_total:.2f} s for {num_frames} frames ({num_frames / time_total:.2f} Hz)")
    if mot_accums:
        evaluate_mot_accums(mot_accums, [str(s.seq_name) for s in dataset if not s.no_gt], generate_overall=True)
