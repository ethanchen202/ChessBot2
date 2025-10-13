import chess
import chess.pgn
import numpy as np
from collections import deque
import h5py
import pickle
import lmdb

from run_timer import TIMER

games_processed = 0

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
# SLIDING_OFFSETS = {
#     "N": 8, "S": -8, "E": 1, "W": -1, "NE": 9, "NW": 7, "SE": -7, "SW": -9
# }

# Knight moves: (row_offset, col_offset)
KNIGHT_OFFSETS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1)
]

# Pawn promotions: straight, capture left, capture right
# PROMOTION_OFFSETS = [
#     ("N", None), ("NE", chess.QUEEN), ("NW", chess.QUEEN),
#     ("NE", chess.ROOK), ("NW", chess.ROOK),
#     ("NE", chess.BISHOP), ("NW", chess.BISHOP),
#     ("NE", chess.KNIGHT), ("NW", chess.KNIGHT)
# ]
PROMOTION_OFFSETS = [
    ("N", chess.ROOK), ("NE", chess.ROOK), ("NW", chess.ROOK),
    ("N", chess.KNIGHT), ("NE", chess.KNIGHT), ("NW", chess.KNIGHT),
    ("N", chess.BISHOP), ("NE", chess.BISHOP), ("NW", chess.BISHOP),
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
    return planes if board.turn == chess.WHITE else np.flip(planes, axis=1)


def encode_board(board, history):
    """
    Encode a single board into AlphaZero-style input tensors.
    """
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
        if board.turn == chess.BLACK:
            ep_plane = np.flip(ep_plane, axis=1)

    # Concat
    input_tensor = np.concatenate([stacked_planes, side_plane, castle_plane_kw, 
                        castle_plane_qw, castle_plane_kb, castle_plane_qb, ep_plane], axis=0)

    return input_tensor


def move_to_index(move, board):
    """
    Map a chess.Move to AlphaZero-style index (0-4671)
    """
    def square_to_rowcol(square):
        return divmod(square, 8)

    from_sq = move.from_square
    to_sq = move.to_square
    from_row, from_col = square_to_rowcol(from_sq)
    to_row, to_col = square_to_rowcol(to_sq)

    piece = board.piece_at(from_sq)
    if piece is None:
        return None

    # Pawn promotions (excluding queen promotions, which stay in sliding category)
    if piece.piece_type == chess.PAWN and move.promotion and move.promotion != chess.QUEEN:
        dr = to_row - from_row
        dc = to_col - from_col
        # Match against PROMOTION_OFFSETS
        for i, (direction, prom_piece) in enumerate(PROMOTION_OFFSETS):
            if move.promotion == prom_piece:
                if direction == "N" and dc == 0:
                    move_type = 64 + i
                    break
                elif direction == "NE" and dr != 0 and dc > 0:
                    move_type = 64 + i
                    break
                elif direction == "NW" and dr != 0 and dc < 0:
                    move_type = 64 + i
                    break
        else:
            raise ValueError("Invalid promotion move")

    # Sliding pieces (rook, bishop, queen, king, pawn non-promotion pushes)
    elif piece.piece_type in [chess.ROOK, chess.BISHOP, chess.QUEEN, chess.KING, chess.PAWN]:
        dr = to_row - from_row
        dc = to_col - from_col
        if dr == 0:  # horizontal
            direction = 3 if dc < 0 else 2
            steps = abs(dc) - 1
        elif dc == 0:  # vertical
            direction = 1 if dr < 0 else 0
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
        else:
            raise ValueError("Invalid sliding move")
        move_type = direction * 7 + steps

    # Knight moves
    elif piece.piece_type == chess.KNIGHT:
        dr = to_row - from_row
        dc = to_col - from_col
        for i, (kdr, kdc) in enumerate(KNIGHT_OFFSETS):
            if (dr, dc) == (kdr, kdc):
                move_type = 56 + i
                break
        else:
            raise ValueError("Invalid knight move")
    else:
        raise ValueError("Unknown piece type")

    return from_sq * NUM_MOVE_TYPES + move_type


def index_to_move(index, board):
    """
    Inverse of move_to_index:
    Convert policy index (0..4671) back into a chess.Move.
    """
    from_sq = index // NUM_MOVE_TYPES
    move_type = index % NUM_MOVE_TYPES
    from_row, from_col = divmod(from_sq, 8)

    # Sliding moves
    if move_type < 56:
        direction = move_type // 7
        steps = (move_type % 7) + 1
        dir_map = {
            0: (1, 0),   # N
            1: (-1, 0),  # S
            2: (0, 1),   # E
            3: (0, -1),  # W
            4: (1, 1),   # NE
            5: (1, -1),  # NW
            6: (-1, 1),  # SE
            7: (-1, -1)  # SW
        }
        dr, dc = dir_map[direction]
        to_row = from_row + dr * steps
        to_col = from_col + dc * steps
        to_sq = to_row * 8 + to_col
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN and to_row in [0, 7]:
            return chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        return chess.Move(from_sq, to_sq)

    # Knight moves
    elif move_type < 64:
        i = move_type - 56
        dr, dc = KNIGHT_OFFSETS[i]
        to_row = from_row + dr
        to_col = from_col + dc
        to_sq = to_row * 8 + to_col
        return chess.Move(from_sq, to_sq)

    # Promotions (64+)
    else:
        i = move_type - 64
        direction, prom_piece = PROMOTION_OFFSETS[i]
        piece = board.piece_at(from_sq)
        if piece is None:
            raise ValueError("Promotion move from empty square")

        dr = 1 if piece.color == chess.WHITE else -1
        if direction == "N":
            to_row = from_row + dr
            to_col = from_col
        elif direction == "NE":
            to_row = from_row + dr
            to_col = from_col + 1
        elif direction == "NW":
            to_row = from_row + dr
            to_col = from_col - 1
        else:
            raise ValueError(f"Invalid promotion direction {direction}")

        to_sq = to_row * 8 + to_col
        return chess.Move(from_sq, to_sq, promotion=prom_piece)


def move_to_policy_vector(move, board):
    index = move_to_index(move, board)
    vec = np.zeros(POLICY_VECTOR_SIZE, dtype=np.float32)
    vec[index] = 1.0
    return vec


def policy_vector_to_move(tensor, board):
    index = np.argmax(tensor)
    move = index_to_move(index.item(), board)
    return move


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
        
        input_tensor = encode_board(board, history)

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
    global games_processed

    # all_games_tensors = []
    # all_policy_tensors = []
    # all_value_tensors = []

    with open(pgn_path, "r") as pgn_file:
        game = chess.pgn.read_game(pgn_file)
        skip = games_processed
        TIMER.start("Skipping already encoded games")
        for _ in range(skip):
            game = chess.pgn.read_game(pgn_file)
        TIMER.stop("Skipping already encoded games")
        for chunk in range(num_games // chunk_size):
            all_games_tensors = []
            all_policy_tensors = []
            all_value_tensors = []
            for _ in range(chunk_size):
                if game is None:
                    break
                game_tensors, policy_tensors, value_tensors = encode_game(game, history_length=history_length)
                all_games_tensors.append(game_tensors)
                all_policy_tensors.append(policy_tensors)
                all_value_tensors.append(value_tensors)
                game = chess.pgn.read_game(pgn_file)

            games_processed += 1
            yield np.concatenate(all_games_tensors, axis=0), np.concatenate(all_policy_tensors, axis=0), np.concatenate(all_value_tensors, axis=0)


def store_h5py(pgn_path, h5py_path, num_games=2173847, max_samples=1000, history_length=1, chunk_size=50):
    """
    Preprocesses the given pgn dataset and stores it in h5 format.
    """
    with h5py.File(h5py_path, "w") as f:

        TIMER.start("Creating datasets")
        dset_x = f.create_dataset("features", shape=(max_samples, 18, 8, 8), dtype="float32", chunks=None, compression=None)
        dset_p = f.create_dataset("policies", shape=(max_samples, 4672), dtype="float32", chunks=None, compression=None)
        dset_v = f.create_dataset("values", shape=(max_samples,), dtype="float32", chunks=None, compression=None)
        TIMER.stop("Creating datasets")
        
        current_size = 0
        TIMER.start(f"Data sample {current_size}/{max_samples}")
        for tensor, policy, value in encode_pgn_file(pgn_path, history_length, num_games, chunk_size):

            TIMER.stop(f"Data sample {current_size}/{max_samples}")
            old_size = current_size
            new_size = old_size + tensor.shape[0]
            TIMER.start(f"Data sample {new_size}/{max_samples}")

            if new_size > max_samples:
                new_size = max_samples
                tensor = tensor[:(new_size - old_size)]
                policy = policy[:(new_size - old_size)]
                value = value[:(new_size - old_size)]
            
            dset_x[old_size:new_size] = tensor
            dset_p[old_size:new_size] = policy
            dset_v[old_size:new_size] = value

            current_size = new_size
            if current_size >= max_samples:
                break

        f.attrs["num_samples"] = current_size
        f.attrs["history_length"] = history_length



def store_lmdb(pgn_path, lmdb_path, num_games=2173847, max_samples=1000, history_length=1, chunk_size=50):
    """
    Preprocesses the PGN dataset and stores it in LMDB format.
    Each sample is serialized with pickle and stored individually.
    """
    TIMER.start("Creating LMDB")
    # Estimate map_size: very rough estimate (e.g., 1 GB per 100k samples)
    map_size = max_samples * (18*8*8 + 4672 + 1) * 4 * 10  # float32, times 2 safety factor

    env = lmdb.open(lmdb_path, map_size=map_size)

    current_size = 0
    progress = 0
    TIMER.start(f"Writing data")
    
    with env.begin(write=True) as txn:
        for tensor, policy, value in encode_pgn_file(pgn_path, history_length, num_games, chunk_size):
            for i in range(tensor.shape[0]):
                if current_size >= max_samples:
                    break

                # Serialize sample as a tuple
                sample = (tensor[i], policy[i], value[i])
                key = f"{current_size:08}".encode("ascii")
                txn.put(key, pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))

                current_size += 1

            if current_size / max_samples > progress:
                TIMER.lap("Writing data", current_size, max_samples)
                print(f"Currently: {current_size}/{max_samples}")
                progress += 0.01

            if current_size >= max_samples:
                break

        # Store metadata as a special key
        txn.put(b"__len__", pickle.dumps(current_size))
        txn.put(b"__history_length__", pickle.dumps(history_length))

    env.close()
    print(f"Finished writing {current_size} samples to LMDB")



if __name__ == "__main__":
    data_path = r'/teamspace/studios/this_studio/chess_bot/datasets/raw/CCRL-4040/CCRL-4040.[2173847].pgn'
    h5py_path = r'/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040.h5'
    lmdb_path_train = r'/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040-train-2m-100k.lmdb'
    lmdb_path_val = r'/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040-val-2m-100k.lmdb'

    store_lmdb(pgn_path=data_path, lmdb_path=lmdb_path_train, max_samples=2000000, history_length=1, chunk_size=5)
    store_lmdb(pgn_path=data_path, lmdb_path=lmdb_path_val, max_samples=100000, history_length=1, chunk_size=5)
