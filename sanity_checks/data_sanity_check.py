import sys

sys.path.append("/teamspace/studios/this_studio/chess_bot/src")

from dataloader import CCRL4040LMDBDataset, worker_init_fn  # type: ignore
from run_timer import TIMER                                 # type: ignore
from torch.utils.data import DataLoader
import numpy as np
import chess

import numpy as np
import chess


piece_to_index = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

# Map from plane index to piece symbol
index_to_piece = {
    0: "P",  # White pawn
    1: "N",
    2: "B",
    3: "R",
    4: "Q",
    5: "K",
    6: "p",
    7: "n",
    8: "b",
    9: "r",
    10: "q",
    11: "k",
}

def print_board(tensor):
    board_str = ""
    for rank in range(8):  # ranks 0..7
        row_str = ""
        for file in range(8):  # files 0..7
            piece_symbol = "."
            for plane in range(12):
                if tensor[plane, rank, file] == 1:
                    piece_symbol = index_to_piece[plane]
                    break
            row_str += piece_symbol + " "
        board_str += row_str + "\n"
    print(board_str)



if __name__ == "__main__":
    dataset = CCRL4040LMDBDataset("/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040.lmdb")


    TIMER.start("Initializing Dataloader")
    h5_path = "/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040.h5"
    lmdb_path = "/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040.lmdb"

    dataset = CCRL4040LMDBDataset(lmdb_path)

    dataloader = DataLoader(
        CCRL4040LMDBDataset(lmdb_path),
        batch_size=320,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        persistent_workers=True
    )
    TIMER.stop("Initializing Dataloader")

    # TIMER.start("Loading data batch")
    for x, policy, value in dataloader:
        # TIMER.stop("Loading data batch")
        print(x.shape, policy.shape, value.shape)
        print_board(x[0, :12, :, :])
        breakpoint()
        # TIMER.start("Loading data batch")
    
    print(dataset[0])