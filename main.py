import argparse
import os
import sys

import numpy as np

import datetime

import time
import torch
import torchvision
import pytorch_lightning as pl


from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer


# from pytorch_lightning.utilities.rank_zero  import rank_zero_only
# from pytorch_lightning.utilities import rank_zero_info

from dataloader import DataModeuleFromConfig


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--name", type=str, const=True, default="", help="postfix for logdir")
    parser.add_argument("--resume", type=str, const=True, default="", help="resume from logdir or checkpoint in logdir")
    parser.add_argument("--base", nargs="*", metavar="base_config.yaml", help="paths to base configs. Loaded from left-to-right.", default=list())
    parser.add_argument("--train", type=str2bool, const=True, default=True, help="train model")
    parser.add_argument("--no-test", type=str2bool, const=True, default=False, help="disable test")
    parser.add_argument("--seed", type=int, default=23, help="seed for seed_everything")
    parser.add_argument("--logdir", type=str, default="logs", help="directory to save logs")
    parser.add_argument("--scale_lr", type=str2bool, const=True, default=True, help="scale base-lr by ngpu * batch_size * n_accumulate")


if __name__=="__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")




