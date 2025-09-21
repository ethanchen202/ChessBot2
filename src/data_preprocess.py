import chess
import chess.pgn
import numpy as np
from collections import deque
import h5py


# Map piece type to plane index
piece_to_index = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

NUM_SQUARES = 64
NUM_MOVE_TYPES = 73
POLICY_VECTOR_SIZE = NUM_SQUARES * NUM_MOVE_TYPES  # 4672

# Directions for sliding pieces (N, S, E, W, NE, NW, SE, SW)
SLIDING_OFFSETS = {
    "N": 8, "S": -8, "E": 1, "W": -1, "NE": 9, "NW": 7, "SE": -7, "SW": -9
}

# Knight moves: (row_offset, col_offset)
KNIGHT_OFFSETS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1)
]

# Pawn promotions: straight, capture left, capture right
PROMOTION_OFFSETS = [
    ("N", None), ("NE", chess.QUEEN), ("NW", chess.QUEEN),
    ("NE", chess.ROOK), ("NW", chess.ROOK),
    ("NE", chess.BISHOP), ("NW", chess.BISHOP),
    ("NE", chess.KNIGHT), ("NW", chess.KNIGHT)
]

# Value mapping
VALUE_MAP = {
    "1-0": 1,
    "1/2-1/2": 0,
    "0-1": -1,
}



def board_to_planes(board):
    """
    Convert a single board to 12x8x8 planes (AlphaZero style).
    """
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


def move_to_index(move, board):
    """
    Map a chess.Move to AlphaZero-style index (0-4671)
    """
    def square_to_rowcol(square):
        """Convert 0-63 square to (row, col)"""
        return divmod(square, 8)

    def rowcol_to_square(row, col):
        """Convert (row, col) to 0-63 square"""
        return row * 8 + col

    from_sq = move.from_square
    to_sq = move.to_square
    from_row, from_col = square_to_rowcol(from_sq)
    to_row, to_col = square_to_rowcol(to_sq)

    # Sliding pieces
    piece = board.piece_at(from_sq)
    if piece == None:
        return

    if piece.piece_type in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
        dr = to_row - from_row
        dc = to_col - from_col
        # Determine direction
        if dr == 0:  # horizontal
            direction = 3 if dc < 0 else 2  # W or E
            steps = abs(dc) - 1
        elif dc == 0:  # vertical
            direction = 1 if dr < 0 else 0  # S or N
            steps = abs(dr) - 1
        elif abs(dr) == abs(dc):  # diagonal
            if dr > 0 and dc > 0:
                direction = 4  # NE
            elif dr > 0 and dc < 0:
                direction = 5  # NW
            elif dr < 0 and dc > 0:
                direction = 6  # SE
            else:
                direction = 7  # SW
            steps = abs(dr) - 1
        move_type = direction * 7 + steps
    # Knight moves
    elif piece.piece_type == chess.KNIGHT:
        dr = to_row - from_row
        dc = to_col - from_col
        for i, (kdr, kdc) in enumerate(KNIGHT_OFFSETS):
            if (dr, dc) == (kdr, kdc):
                move_type = 56 + i
                break
    # Pawn promotions
    elif piece.piece_type == chess.PAWN and move.promotion:
        for i, (_, prom_piece) in enumerate(PROMOTION_OFFSETS):
            if move.promotion == prom_piece:
                move_type = 64 + i
                break
    # Normal pawn moves (no promotion)
    elif piece.piece_type == chess.PAWN:
        # treat as sliding pawn forward moves (1-7 squares)
        dr = to_row - from_row
        if piece.color == chess.BLACK:
            dr = -dr
        move_type = 64 + dr - 1  # approximate
    # King moves (can be treated like single-step sliding)
    elif piece.piece_type == chess.KING:
        dr = to_row - from_row
        dc = to_col - from_col
        # single step sliding
        direction_map = {(1,0):0, (-1,0):1, (0,1):2, (0,-1):3, (1,1):4, (1,-1):5, (-1,1):6, (-1,-1):7}
        move_type = direction_map.get((dr,dc), 0)  # simplification
    else:
        raise ValueError("Unknown piece type")

    index = from_sq * NUM_MOVE_TYPES + move_type
    return index


def move_to_policy_vector(move, board):
    index = move_to_index(move, board)
    vec = np.zeros(POLICY_VECTOR_SIZE, dtype=np.float32)
    vec[index] = 1.0
    return vec


def encode_game(game, history_length=8):
    """
    Encode a single chess game into AlphaZero-style input tensors.
    
    Returns a list of tensors: one per move.
    """
    board = game.board()
    history = deque(maxlen=history_length)
    result = VALUE_MAP[game.headers['Result']]
    tensors = []
    policies = []
    values = []

    # Fill initial history with empty boards
    empty_planes = np.zeros((12, 8, 8), dtype=np.float32)
    for _ in range(history_length):
        history.append(empty_planes)

    for move in game.mainline_moves():
        current_planes = board_to_planes(board)
        history.append(current_planes)

        # Stack history
        stacked_planes = np.concatenate(list(history), axis=0)

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

        # Concat
        input_tensor = np.concatenate([stacked_planes, side_plane, castle_plane_kw, 
                            castle_plane_qw, castle_plane_kb, castle_plane_qb, ep_plane], axis=0)
        

        # Orient board based on current player
        if board.turn == chess.BLACK:
            input_tensor = np.flip(input_tensor, axis=1)

        tensors.append(input_tensor)
        policies.append(move_to_policy_vector(move, board))
        values.append(result)
        board.push(move)

    return np.array(tensors), np.array(policies), np.array(values)


def encode_pgn_file(pgn_path, history_length=8, num_games=1000, chunk_size=100):
    """
    Encode all games in a PGN file.
    
    Returns a list of games, where each game is a list of tensors.
    """
    all_games_tensors = []
    all_policy_tensors = []
    all_value_tensors = []

    with open(pgn_path, "r") as pgn_file:
        game = chess.pgn.read_game(pgn_file)
        for chunk in range(num_games // chunk_size):
            for _ in range(chunk_size):
                if game is None:
                    break
                game_tensors, policy_tensors, value_tensors = encode_game(game, history_length=history_length)
                all_games_tensors.append(game_tensors)
                all_policy_tensors.append(policy_tensors)
                all_value_tensors.append(value_tensors)
                game = chess.pgn.read_game(pgn_file)

            yield np.concatenate(all_games_tensors, axis=0), np.concatenate(all_policy_tensors, axis=0), np.concatenate(all_value_tensors, axis=0)


def store_h5py(pgn_path, h5py_path, num_games=1000, history_length=1, chunk_size=500):
    """
    Preprocesses the given pgn dataset and stores it in h5 format.
    """
    with h5py.File(h5py_path, "w") as f:
        dset_x = f.create_dataset("features", shape=(0, 18, 8, 8), maxshape=(None, 18, 8, 8), dtype="float32", chunks=True)
        dset_p = f.create_dataset("policies", shape=(0, 4672), maxshape=(None, 4672), dtype="float32", chunks=True)
        dset_v = f.create_dataset("values", shape=(0,), maxshape=(None,), dtype="float32", chunks=True)
        
        for tensor, policy, value in encode_pgn_file(pgn_path, history_length, num_games, chunk_size):
            old_size = dset_x.shape[0]
            new_size = old_size + tensor.shape[0]
            
            dset_x.resize(new_size, axis=0)
            dset_p.resize(new_size, axis=0)
            dset_v.resize(new_size, axis=0)
            
            dset_x[old_size:new_size] = tensor
            dset_p[old_size:new_size] = policy
            dset_v[old_size:new_size] = value


if __name__ == "__main__":
    data_path = r'/teamspace/studios/this_studio/chess_bot/datasets/raw/CCRL-4040/CCRL-4040.[2173847].pgn'
    h5py_path = r'/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040.h5'

    # tensor, policy, value = encode_pgn_file(file_path, history_length=1, num_games=1)
    store_h5py(pgn_path=data_path, h5py_path=h5py_path, num_games=1000, history_length=1, chunk_size=100)
