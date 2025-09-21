import chess
import chess.pgn
import numpy as np
from collections import deque

# Map piece type to plane index
piece_to_index = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

def board_to_planes(board):
    """Convert a single board to 12x8x8 planes (AlphaZero style)."""
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            idx = piece_to_index[piece.piece_type]
            if piece.color == chess.BLACK:
                idx += 6
            planes[idx, row, col] = 1
    return planes

def encode_game(game, history_length=8):
    """
    Encode a single chess game into AlphaZero-style input tensors.
    
    Returns a list of tensors: one per move.
    """
    board = game.board()
    history = deque(maxlen=history_length)
    tensors = []

    # Fill initial history with empty boards
    empty_planes = np.zeros((12, 8, 8), dtype=np.float32)
    for _ in range(history_length):
        history.append(empty_planes)

    for move in game.mainline_moves():
        current_planes = board_to_planes(board)
        history.append(current_planes)

        # Stack history
        stacked_planes = np.concatenate(list(history), axis=0)

        # Side-to-move plane
        side_plane = np.full((1, 8, 8), int(board.turn), dtype=np.float32)

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

        # Concat
        input_tensor = np.concatenate([stacked_planes, side_plane, castle_plane_kw, 
                            castle_plane_qw, castle_plane_kb, castle_plane_qb, ep_plane], axis=0)
        

        # Orient board based on current player
        if board.turn == chess.BLACK:
            input_tensor = np.flip(input_tensor, axis=1)

        tensors.append(input_tensor)
        board.push(move)

    return np.array(tensors)

def encode_pgn_file(pgn_path, history_length=8, num_games=1000):
    """
    Encode all games in a PGN file.
    
    Returns a list of games, where each game is a list of tensors.
    """
    all_games_tensors = []

    with open(pgn_path, "r") as pgn_file:
        game = chess.pgn.read_game(pgn_file)
        for _ in range(num_games):
            if game is None:
                break
            game_tensors = encode_game(game, history_length=history_length)
            all_games_tensors.append(game_tensors)
            game = chess.pgn.read_game(pgn_file)

    return np.concatenate(all_games_tensors, axis=0)

file_path = r'/teamspace/studios/this_studio/chess_bot/datasets/raw/CCRL-4040/CCRL-4040.[2173847].pgn'

tensor = encode_pgn_file(file_path, history_length=1, num_games=10)

breakpoint()
