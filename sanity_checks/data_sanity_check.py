import sys

sys.path.append("/teamspace/studios/this_studio/chess_bot/src")

from data_preprocess import move_to_index, index_to_move    # type: ignore
from dataloader import CCRL4040LMDBDataset, worker_init_fn  # type: ignore
from run_timer import TIMER                                 # type: ignore
from torch.utils.data import DataLoader
import numpy as np
import chess
import chess.pgn


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


def sanity_check_input(h5_path, lmdb_path):
    dataset = CCRL4040LMDBDataset(lmdb_path)

    TIMER.start("Initializing Dataloader")

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


def sanity_check_policy(pgn_path, num_games=100):
    with open(pgn_path, "r") as pgn_file:
        TIMER.start("Reading game")
        correct = 0
        total = 0
        for game_idx in range(num_games):
            game = chess.pgn.read_game(pgn_file)
            board = game.board()                    # type: ignore
            for move in game.mainline_moves():      # type: ignore
                piece = board.piece_at(move.from_square)
                try:
                    if move == index_to_move(move_to_index(move, board), board):
                        correct += 1
                    else:
                        print(board)
                        print(piece, end=" | ")
                        print(move, end=" | ")
                        print(move_to_index(move, board), end=" | ")
                        print(index_to_move(move_to_index(move, board), board), end=" | ")
                        print(move == index_to_move(move_to_index(move, board), board))
                except ValueError:
                    print(board)
                    breakpoint()
                    print(piece, end=" | ")
                    print(move, end=" | ")
                    print(move_to_index(move, board), end=" | ")
                    print(index_to_move(move_to_index(move, board), board), end=" | ")
                    print(move == index_to_move(move_to_index(move, board), board))
                total += 1
                board.push(move)
            TIMER.lap("Reading game", game_idx + 1, num_games)
        TIMER.stop("Reading game")
        print(correct, total, correct / total)
            



if __name__ == "__main__":
    h5_path = r"/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040.h5"
    lmdb_path = r"/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040.lmdb"
    pgn_path = r"/teamspace/studios/this_studio/chess_bot/datasets/raw/CCRL-4040/CCRL-4040.[2173847].pgn"
    # sanity_check_input(h5_path, lmdb_path)
    sanity_check_policy(pgn_path, num_games=1000)