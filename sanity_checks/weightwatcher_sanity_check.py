import sys

sys.path.append("/teamspace/studios/this_studio/chess_bot/src")

from data_sanity_check import print_board, index_to_piece # type: ignore
from model import ChessViT # type: ignore
from model2 import ChessViTv2 # type: ignore
from run_timer import TIMER # type: ignore
from data_preprocess import index_to_move, HISTORY_LEN, encode_board, get_legal_mask # type: ignore

from torch.cuda import is_available
import weightwatcher as ww
import torch
import numpy as np
import chess
from collections import deque
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

if __name__ == "__main__":
    with torch.no_grad():
        TIMER.start("Initializing")
        load_from_checkpoint_path = r"/teamspace/studios/this_studio/chess_bot/results/checkpoints/run2-10000-1000-0.2-0.8-1/model_epoch_200.pt"

        # if torch.cuda.is_available():
        #     device = "cuda"
        # else:
        #     device = "cpu"
        device = "cpu"

        model = ChessViTv2()
        model.load_state_dict(torch.load(load_from_checkpoint_path, map_location=torch.device('cpu')))
        TIMER.stop("Initializing")
        
        model.to(device)
        watcher = ww.WeightWatcher(model)
        print(watcher.analyze())