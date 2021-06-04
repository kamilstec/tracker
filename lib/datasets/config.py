import os
import os.path as osp
from easydict import EasyDict as edict


cfg = edict()

cfg.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..', 'gdrive/MyDrive/KamMOT_trenowanie'))
cfg.DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', 'data'))

def get_data_dir(module):
    outdir = osp.abspath(osp.join(cfg.DATA_DIR, module))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def get_output_dir(module):
    outdir = osp.abspath(osp.join(cfg.ROOT_DIR, 'output', module))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def get_tb_dir(module):
    outdir = osp.abspath(osp.join(cfg.ROOT_DIR, 'tensorboard', module))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir
