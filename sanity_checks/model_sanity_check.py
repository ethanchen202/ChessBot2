import sys

from torch.cuda import is_available

sys.path.append("/teamspace/studios/this_studio/chess_bot/src")

from data_sanity_check import print_board, index_to_piece
from model import ChessViT # type: ignore
from run_timer import TIMER # type: ignore
from data_preprocess import policy_vector_to_move, board_to_planes, index_to_move # type: ignore
import torch
import numpy as np
import chess


import chess

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
        checkpoint_dir = r"/teamspace/studios/this_studio/chess_bot/results/checkpoints/dataset-1m_lr-1e-4/model_epoch_40.pt"

        # if torch.cuda.is_available():
        #     device = "cuda"
        # else:
        #     device = "cpu"
        device = "cpu"

        model = ChessViT()
        model.load_state_dict(torch.load(checkpoint_dir, map_location=torch.device('cpu')))

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

        if op_move:
            board.push(chess.Move.from_uci(op_move))

        while not board.is_game_over():        

            TIMER.start("forward pass")

            # Stack history
            current_planes = board_to_planes(board)

            # Side-to-move plane (1/-1 for white/black)
            side_plane = np.full((1, 8, 8), (1 if board.turn == chess.WHITE else -1), dtype=np.float32)

            # Castling Rights
            castle_plane_kw = np.full((1, 8, 8), (1 if board.has_kingside_castling_rights(chess.WHITE) else 0), dtype=np.float32)
            castle_plane_qw = np.full((1, 8, 8), (1 if board.has_queenside_castling_rights(chess.WHITE) else 0), dtype=np.float32)
            castle_plane_kb = np.full((1, 8, 8), (1 if board.has_kingside_castling_rights(chess.BLACK) else 0), dtype=np.float32)
            castle_plane_qb = np.full((1, 8, 8), (1 if board.has_queenside_castling_rights(chess.BLACK) else 0), dtype=np.float32)

            # En Passant
            ep_plane = np.zeros((1, 8, 8), dtype=np.float32)
            if board.ep_square is not None:
                x = chess.square_file(board.ep_square)
                y = 7 - chess.square_rank(board.ep_square)
                ep_plane[0, y, x] = 1

            input_tensor = np.concatenate([current_planes, side_plane, castle_plane_kw, 
                                    castle_plane_qw, castle_plane_kb, castle_plane_qb, ep_plane], axis=0)
            input_tensor = torch.tensor(input_tensor).unsqueeze(0)

            print(input_tensor.shape)

            policy_logits, value = model(input_tensor)
            
            # Get probabilities (optional, but makes sorting clear)
            probs = torch.softmax(policy_logits[0], dim=-1)

            # Sort move indices by descending probability
            sorted_indices = torch.argsort(probs, descending=True)

            # Now iterate until we find a legal move
            best_move = None
            for idx in sorted_indices:
                idx = idx.item()
                
                # Convert index to move
                move = index_to_move(idx, board)
                from_piece = board.piece_at(move.from_square)

                if move in board.legal_moves:
                    best_move = move
                    break

            if best_move is None:
                raise ValueError("No legal move found in logits!")  # should not happen

            TIMER.stop("forward pass")

            print(f"Best move: {best_move}")
            print(board)

            board.push(best_move)

            op_move = input("Enter opponent move: ")
            board.push(chess.Move.from_uci(op_move))
