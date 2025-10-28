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


def board_from_ascii(ascii_board):
    """
    Converts an ASCII chessboard string to a python-chess Board object
    by manually placing pieces.
    
    Example input:
    "r.bqk..r
     pppp.ppp
     ..n..n..
     ..b.p...
     ..B.P...
     .....N..
     PPPP.PPP
     RNBQ.RK."
    """
    # Remove leading/trailing whitespace and split into rows
    rows = ascii_board.strip().splitlines()
    
    if len(rows) != 8:
        raise ValueError("ASCII board must have 8 rows")
    
    board = chess.Board.empty()  # Start with empty board
    
    # Map from ASCII chars to chess pieces
    piece_map = {
        'r': chess.ROOK,
        'n': chess.KNIGHT,
        'b': chess.BISHOP,
        'q': chess.QUEEN,
        'k': chess.KING,
        'p': chess.PAWN
    }
    
    # Iterate over rows (from top row to bottom)
    for rank_index, row in enumerate(rows):
        row = row.strip()
        if len(row) != 8:
            raise ValueError(f"Row {rank_index + 1} must have 8 characters")
        
        rank = 7 - rank_index  # chess.Board ranks go from 0 (a1) to 7 (h8)
        for file_index, char in enumerate(row):
            file = file_index
            if char != '.':
                color = chess.WHITE if char.isupper() else chess.BLACK
                piece_type = piece_map[char.lower()]
                board.set_piece_at(chess.square(file, rank), chess.Piece(piece_type, color))
    
    return board




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

        # board =  """r..qkr..
        #             ppp..pp.
        #             ...pbn..
        #             ..bNp..p
        #             ..B.P...
        #             .N.P....
        #             PPP..PPP
        #             R.BQ.RK."""
        # board = board_from_ascii(board)

        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        print(board)

        model.eval()

        op_move = input("Enter opponent move (press enter for White): ")

        history = deque(maxlen=HISTORY_LEN)
        empty_planes = np.zeros((12, 8, 8), dtype=np.float32)
        for _ in range(HISTORY_LEN):
            history.append(empty_planes)

        if op_move:
            input_tensor = encode_board(board, history)
            board.push(chess.Move.from_uci(op_move))

        while not board.is_game_over():

            # breakpoint()

            TIMER.start("forward pass")

            input_tensor = encode_board(board, history)
            input_tensor = torch.as_tensor(input_tensor).unsqueeze(0)
            legal_mask = torch.as_tensor(get_legal_mask(board), dtype=torch.bool)

            policy_logits, value, _, _ = model(input_tensor, legal_mask)
            
            # Get probabilities (optional, but makes sorting clear)
            probs = torch.softmax(policy_logits[0], dim=-1)

            # Sort move indices by descending probability
            sorted_indices = torch.argsort(probs, descending=True)

            # Now iterate until we find a legal move
            best_move = None
            num_iterations = 0
            for idx in sorted_indices:
                num_iterations += 1
                idx = idx.item()
                
                # Convert index to move
                move = index_to_move(idx, board)
                from_piece = board.piece_at(move.from_square)

                if move in board.legal_moves:
                    print(f"Best move found at iteration {num_iterations}")
                    best_move = move
                    break

            if best_move is None:
                raise ValueError("No legal move found in logits!")  # should not happen

            TIMER.stop("forward pass")

            board.push(best_move)

            print(board)
            print(f"Value: {value} | Best move: {best_move}")

            op_move = input("Enter opponent move: ")
            board.push(chess.Move.from_uci(op_move))
