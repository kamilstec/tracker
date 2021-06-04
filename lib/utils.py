import os
from collections import defaultdict
from os import path as osp

import numpy as np
import torch
from cycler import cycler as cy # https://matplotlib.org/cycler/

import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import motmetrics as mm

matplotlib.use('Agg')

colors = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',
    'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
    'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
    'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
    'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
    'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
    'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
    'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
    'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
    'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
    'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
    'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
    'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
    'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]


def plot_sequence(tracks, db, output_dir):

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    cyl = cy('ec', colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    for i, v in enumerate(db):
        im_path = v['img_path']
        im_name = osp.basename(im_path)
        im_output = osp.join(output_dir, im_name)
        im = cv2.imread(im_path)
        im = im[:, :, (2, 1, 0)]

        sizes = np.shape(im)
        height = float(sizes[0])
        width = float(sizes[1])

        fig = plt.figure()
        fig.set_size_inches(width / 100, height / 100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(im)

        for j, t in tracks.items():
            if i in t.keys():
                t_i = t[i]
                ax.add_patch(
                    plt.Rectangle(
                        (t_i[0], t_i[1]),
                        t_i[2] - t_i[0],
                        t_i[3] - t_i[1],
                        fill=False,
                        linewidth=2.0, **styles[j]
                    ))

                ax.annotate(j, xy=(t_i[0], t_i[1]+3),
                            color=styles[j]['ec'], weight='bold', fontsize=20, ha='left', va='bottom')

        plt.axis('off')
        # plt.tight_layout()
        plt.draw()
        plt.savefig(im_output, dpi=100)
        plt.close()


def plot_sequence_ids_frames(ids_frames, tracks, kalman, db, output_dir):

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    for i, v in enumerate(db):
        for choosen_id, choosen_frame in ids_frames:
            if i == choosen_frame:
                im_path = v['img_path']
                im_name = osp.basename(im_path)
                im_output = osp.join(output_dir, im_name)
                im = cv2.imread(im_path)
                im = im[:, :, (2, 1, 0)]

                sizes = np.shape(im)
                height = float(sizes[0])
                width = float(sizes[1])

                fig = plt.figure()
                fig.set_size_inches(width / 100, height / 100)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(im)

                if kalman:
                    colr = 'red'
                else:
                    colr = 'cyan'

                for j, t in tracks.items():
                    if j == choosen_id:
                        if i in t.keys():
                            t_i = t[i]
                            ax.add_patch(
                                plt.Rectangle(
                                    (t_i[0], t_i[1]),
                                    t_i[2] - t_i[0],
                                    t_i[3] - t_i[1],
                                    fill=False,
                                    linewidth=2.0, color=colr
                                ))

                            ax.annotate(j, xy=(t_i[0], t_i[1] + 3),
                                        color=colr, weight='bold', fontsize=20, ha='left', va='bottom')

                plt.axis('off')
                plt.draw()
                plt.savefig(im_output, dpi=100)
                plt.close()

def plot_sequence_ids_frames_kalman(ids_frames, tracks, kalman_tracks, db, output_dir):

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    for i, v in enumerate(db):
        for choosen_id, choosen_frame in ids_frames:
            if i == choosen_frame:
                im_path = v['img_path']
                im_name = osp.basename(im_path)
                im_output = osp.join(output_dir, im_name)
                im = cv2.imread(im_path)
                im = im[:, :, (2, 1, 0)]

                sizes = np.shape(im)
                height = float(sizes[0])
                width = float(sizes[1])

                fig = plt.figure()
                fig.set_size_inches(width / 100, height / 100)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(im)

                # Ramki ścieżek
                for j, t in tracks.items():
                    if j == choosen_id:
                        if i in t.keys():
                            t_i = t[i]
                            ax.add_patch(
                                plt.Rectangle(
                                    (t_i[0], t_i[1]),
                                    t_i[2] - t_i[0],
                                    t_i[3] - t_i[1],
                                    fill=False,
                                    linewidth=2.0, color='cyan'
                                ))

                            ax.annotate(j, xy=(t_i[0], t_i[1] + 3),
                                        color='cyan', weight='bold', fontsize=20, ha='left', va='bottom')

                # Ramki przewidziane przez Kalmana
                for j, t in kalman_tracks.items():
                    if j == choosen_id:
                        if i in t.keys():
                            t_i = t[i]
                            ax.add_patch(
                                plt.Rectangle(
                                    (t_i[0], t_i[1]),
                                    t_i[2] - t_i[0],
                                    t_i[3] - t_i[1],
                                    fill=False,
                                    linewidth=2.0, color='red'
                                ))

                plt.axis('off')
                plt.draw()
                plt.savefig(im_output, dpi=100)
                plt.close()


def plot_tracks(blobs, tracks, gt_tracks=None, output_dir=None, name=None):
    im_paths = blobs['im_paths']
    if not name:
        im0_name = osp.basename(im_paths[0])
    else:
        im0_name = str(name) + ".jpg"
    im0 = cv2.imread(im_paths[0])
    im1 = cv2.imread(im_paths[1])
    im0 = im0[:, :, (2, 1, 0)]
    im1 = im1[:, :, (2, 1, 0)]

    im_scales = blobs['im_info'][0, 2]

    tracks = tracks.data.cpu().numpy() / im_scales
    num_tracks = tracks.shape[0]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(im0, aspect='equal')
    ax[1].imshow(im1, aspect='equal')

    cyl = cy('ec', colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    ax[0].set_title(('{} tracks').format(num_tracks), fontsize=14)

    for i, t in enumerate(tracks):
        t0 = t[0]
        t1 = t[1]
        ax[0].add_patch(
            plt.Rectangle(
                (t0[0], t0[1]),
                t0[2] - t0[0],
                t0[3] - t0[1], fill=False,
                linewidth=1.0, **styles[i]
            ))

        ax[1].add_patch(
            plt.Rectangle(
                (t1[0], t1[1]),
                t1[2] - t1[0],
                t1[3] - t1[1], fill=False,
                linewidth=1.0, **styles[i]
            ))

    if gt_tracks:
        for gt in gt_tracks:
            for i in range(2):
                ax[i].add_patch(
                    plt.Rectangle(
                        (gt[i][0], gt[i][1]),
                        gt[i][2] - gt[i][0],
                        gt[i][3] - gt[i][1], fill=False,
                        edgecolor='blue', linewidth=1.0
                    ))

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    image = None
    if output_dir:
        im_output = osp.join(output_dir, im0_name)
        plt.savefig(im_output)
    else:
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image

def get_mot_accum(results, seq):
    mot_accum = mm.MOTAccumulator(auto_id=True)    

    for i, data in enumerate(seq):
        gt = data['gt']
        gt_ids = []
        if gt:
            gt_boxes = []
            for gt_id, box in gt.items():
                gt_ids.append(gt_id)
                gt_boxes.append(box)
        
            gt_boxes = np.stack(gt_boxes, axis=0)
            gt_boxes = np.stack((gt_boxes[:, 0],
                                 gt_boxes[:, 1],
                                 gt_boxes[:, 2] - gt_boxes[:, 0],
                                 gt_boxes[:, 3] - gt_boxes[:, 1]),
                                axis=1)
        else:
            gt_boxes = np.array([])
        
        track_ids = []
        track_boxes = []
        for track_id, frames in results.items():
            if i in frames:
                track_ids.append(track_id)
                track_boxes.append(frames[i][:4])

        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            track_boxes = np.stack((track_boxes[:, 0],
                                    track_boxes[:, 1],
                                    track_boxes[:, 2] - track_boxes[:, 0],
                                    track_boxes[:, 3] - track_boxes[:, 1]),
                                    axis=1)
        else:
            track_boxes = np.array([])

        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)

        mot_accum.update(
            gt_ids,
            track_ids,
            distance)

    return mot_accum


def evaluate_mot_accums(accums, names, generate_overall=False):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums,
        metrics=['idf1', 'idp', 'idr', 'num_switches', 'mota'],
        names=names,
        generate_overall=generate_overall,)

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'num_switches': 'IDs', 'mota': 'MOTA'})
    print(str_summary)
