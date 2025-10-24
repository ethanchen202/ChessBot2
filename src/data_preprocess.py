import decimal
import chess
import chess.pgn
import numpy as np
from collections import deque
import h5py
import pickle
import lmdb
import random
import os

from numpy.random import shuffle

from run_timer import TIMER


HISTORY_LEN = 1
METADATA_PLANES = 7
TOTAL_PLANES = HISTORY_LEN * 12 + METADATA_PLANES

OPENING_SAMPLE_PROB = 0.2
MIDGAME_SAMPLE_PROB = 0.8
ENDGAME_SAMPLE_PROB = 1


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

    # 50-Move Rule
    halfmove_plane = np.full((1, 8, 8), board.halfmove_clock / 99.0, dtype=np.float32)

    # Concat
    input_tensor = np.concatenate([stacked_planes, side_plane, castle_plane_kw, 
                        castle_plane_qw, castle_plane_kb, castle_plane_qb, ep_plane, halfmove_plane], axis=0)

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
        print(f"Invalid move detected: {move}")
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


def index_to_policy_vector(index):
    vec = np.zeros(POLICY_VECTOR_SIZE, dtype=np.float32)
    vec[index] = 1.0
    return vec


def move_to_policy_vector(move, board):
    index = move_to_index(move, board)
    vec = index_to_policy_vector(index)
    return vec


def policy_vector_to_move(tensor, board):
    index = np.argmax(tensor)
    move = index_to_move(index.item(), board)
    return move


def get_legal_mask(board):
    mask = np.zeros((POLICY_VECTOR_SIZE,), dtype=np.float32)
    for move in board.legal_moves:
        index = move_to_index(move, board)
        mask[index] = 1.0
    return mask

def encode_legal_mask(legal_mask):
    return np.packbits(legal_mask.astype(np.uint8))

def decode_legal_mask(legal_mask):
    return np.unpackbits(legal_mask.astype(np.uint8))[:4672]


def encode_input_tensor(input_tensor):
    if input_tensor.shape != (TOTAL_PLANES, 8, 8):
        raise ValueError("Input tensor must have shape (19,8,8)")

    # board planes
    binary_planes = input_tensor[:(HISTORY_LEN * 12)].astype(np.uint64)  # ensure 0/1
    binary_packed = (binary_planes.reshape((HISTORY_LEN * 12), 64) * (1 << np.arange(64, dtype=np.uint64))).sum(axis=1)  # shape (12*hl,)

    # metadata planes
    metadata_planes = input_tensor[(HISTORY_LEN * 12):(HISTORY_LEN * 12 + 5)]
    metadata_values = metadata_planes[:, 0, 0].astype(np.int8)  # shape (5,)

    # enpassant plane
    enpassant_plane = input_tensor[(HISTORY_LEN * 12 + 5)]
    enpassant_values = np.packbits(enpassant_plane.astype(np.uint8)) #  shape (1,)

    # halfmove plane
    halfmove_plane = input_tensor[(HISTORY_LEN * 12 + 6)]
    halfmove_values = halfmove_plane[0, 0].astype(np.float32) #  shape (1,)

    return binary_packed, metadata_values, enpassant_values, halfmove_values


def decode_input_tensor(binary_packed, metadata_values, enpassant_values, halfmove_values):
    input_tensor = np.zeros((TOTAL_PLANES, 8, 8), dtype=np.float32)

    # board planes
    mask = (1 << np.arange(64, dtype=np.uint64))
    input_tensor[:(HISTORY_LEN * 12)] = ((binary_packed[:, None] & mask) != 0).astype(np.float32).reshape(12, 8, 8)

    # metadata planes
    input_tensor[(HISTORY_LEN * 12):(HISTORY_LEN * 12 + 5)] = metadata_values.reshape(5, 1, 1).astype(np.float32)

    # enpassant
    input_tensor[(HISTORY_LEN * 12 + 5)] = np.unpackbits(np.frombuffer(enpassant_values, dtype=np.uint8))[:64].reshape(8, 8).astype(np.float32)

    # halfmove planes
    input_tensor[(HISTORY_LEN * 12 + 6)] = halfmove_values.reshape(1, 1, 1).astype(np.float32)

    return input_tensor


def verify_input_tensor_encoding(binary_packed, metadata_values, enpassant_values, halfmove_values, input_tensor):
    """Verify bijective input encoding"""
    test_input_tensor = decode_input_tensor(binary_packed, metadata_values, enpassant_values, halfmove_values)
    encoded_binary_packed, encoded_metadata_values, encoded_enpassant_values, encoded_halfmove_values = encode_input_tensor(input_tensor)
    assert np.array_equal(binary_packed, encoded_binary_packed)
    assert np.array_equal(metadata_values, encoded_metadata_values)
    assert np.array_equal(enpassant_values, encoded_enpassant_values)
    assert np.array_equal(halfmove_values, encoded_halfmove_values)
    assert np.array_equal(input_tensor, test_input_tensor)


def verify_policy_encoding(policy_idx, move, board):
    """Verify bijective policy encoding"""
    test_policy_idx = move_to_index(move, board)
    test_move = index_to_move(policy_idx, board)
    assert test_policy_idx == policy_idx
    assert move.from_square == test_move.from_square and move.to_square == test_move.to_square


def verify_legal_mask_encoding(legal_mask, encoded_legal_mask):
    """Verify bijective legal mask encoding"""
    decoded_legal_mask = decode_legal_mask(encoded_legal_mask)
    test_encoded_mask = encode_legal_mask(legal_mask)
    assert np.array_equal(legal_mask, decoded_legal_mask)
    assert np.array_equal(encoded_legal_mask, test_encoded_mask)


def encode_game(game, history_length=8):
    """
    Encode a single chess game into AlphaZero-style input tensors.
    
    Returns a list of tensors: one per move.
    """
    board = game.board()
    history = deque(maxlen=history_length)
    result = VALUE_MAP[game.headers['Result']]
    board_arrays = [] # (num_samples, history_length * 12) => history_length * 12 uint64 bitmaps for 12x8x8 board tensor
    metadata_arrays = [] # (num_samples, 5) => 5 int8 bitmaps for metadata tensor
    enpassant_arrays = [] # (num_samples,) => 1 uint8 bitmap for enpassant tensor
    num_halfmoves = [] # (num_samples,) => 1 float32 for side to move
    policy_idxs = [] # (num_samples,) => 1 uint16 for policy idx
    values = [] # (num_samples,) => 1 int8 value
    legal_masks = [] # (num_samples, 584) => 584 uint8 bitmaps for legal move tensor
    mv_count = 0

    # Fill initial history with empty boards
    empty_planes = np.zeros((12, 8, 8), dtype=np.float32)
    for _ in range(history_length):
        history.append(empty_planes)

    for move in game.mainline_moves():

        mv_count += 1
        
        input_tensor = encode_board(board, history)
        policy = move_to_index(move, board)
        legal_mask = get_legal_mask(board)

        if mv_count < 15:
            if random.random() > OPENING_SAMPLE_PROB:
                board.push(move)
                continue
        elif mv_count < 40:
            if random.random() > MIDGAME_SAMPLE_PROB:
                board.push(move)
                continue
        else:
            if random.random() > ENDGAME_SAMPLE_PROB:
                board.push(move)
                continue

        board_planes, metadata, enpassant, halfmoves = encode_input_tensor(input_tensor)
        encoded_legal_mask = encode_legal_mask(legal_mask)
        board_arrays.append(board_planes)
        metadata_arrays.append(metadata)
        enpassant_arrays.append(enpassant)
        num_halfmoves.append(halfmoves)
        policy_idxs.append(policy)
        values.append(result)
        legal_masks.append(encoded_legal_mask)

        # Verify board encoding
        if not np.array_equal(decode_input_tensor(board_planes, metadata, enpassant, halfmoves), input_tensor):
            print("Encoding failed")
            decoded = decode_input_tensor(board_planes, metadata, enpassant, halfmoves)
            diff_mask = decoded != input_tensor
            indices = np.argwhere(diff_mask)
            indices = [tuple(idx) for idx in indices]
            raise ValueError(f"Encoding failed at indices {indices}")
        
        # Verify safe data preprocessing
        verify_input_tensor_encoding(board_planes, metadata, enpassant, halfmoves, input_tensor)
        verify_policy_encoding(policy, move, board)
        verify_legal_mask_encoding(legal_mask, encoded_legal_mask)

        # Update board
        board.push(move)

    return np.array(board_arrays, dtype=np.uint64), np.array(metadata_arrays, dtype=np.int8), np.array(enpassant_arrays, dtype=np.uint8), \
        np.array(num_halfmoves, dtype=np.float32),  np.array(policy_idxs, dtype=np.uint16), np.array(values, dtype=np.int8), \
            np.array(legal_masks, dtype=np.uint8) 


def encode_pgn_file(pgn_path, history_length=8, num_games=1000, chunk_size=100):
    """
    Encode all games in a PGN file.
    
    Returns a list of games, where each game is a list of tensors.
    """
    global games_processed

    with open(pgn_path, "r") as pgn_file:
        game = chess.pgn.read_game(pgn_file)
        skip = games_processed
        TIMER.start("Skipping already encoded games")
        for _ in range(skip):
            game = chess.pgn.read_game(pgn_file)
        TIMER.stop("Skipping already encoded games")
        for chunk in range(num_games // chunk_size):
            all_board_tensors = []
            all_metadata_tensors = []
            all_enpassant_tensors = []
            all_halfmoves_tensors = []
            all_policy_tensors = []
            all_value_tensors = []
            all_legal_masks = []
            
            for _ in range(chunk_size):
                if game is None:
                    break
                board_tensors, metadata_tensors, enpassant_tensors, halfmove_tensors, policy_tensors, value_tensors, legal_masks = encode_game(game, history_length=history_length)
                all_board_tensors.append(board_tensors)
                all_metadata_tensors.append(metadata_tensors)
                all_enpassant_tensors.append(enpassant_tensors)
                all_halfmoves_tensors.append(halfmove_tensors)
                all_policy_tensors.append(policy_tensors)
                all_value_tensors.append(value_tensors)
                all_legal_masks.append(legal_masks)
                game = chess.pgn.read_game(pgn_file)

            games_processed += 1
            yield np.concatenate(all_board_tensors, axis=0), np.concatenate(all_metadata_tensors, axis=0), np.concatenate(all_enpassant_tensors, axis=0), \
                np.concatenate(all_halfmoves_tensors, axis=0), np.concatenate(all_policy_tensors, axis=0), np.concatenate(all_value_tensors, axis=0), \
                    np.concatenate(all_legal_masks, axis=0)


def store_lmdb(pgn_path, lmdb_path, num_games=2173847, max_samples=1000, history_length=1, chunk_size=50, shuffle=True):
    """
    Preprocesses the PGN dataset and stores it in LMDB format.
    Each sample is serialized with pickle and stored individually.
    """
    TIMER.start("Creating LMDB")
    # Estimate map_size: very rough estimate (e.g., 1 GB per 100k samples)
    map_size = max_samples * (HISTORY_LEN*12*8 + 5*1 + 8*1 + 1*4 + 1*2 + 1*1 + 4672) * 10  # times 10 safety factor
    print("Preallocating map size: ", map_size)

    env = lmdb.open(lmdb_path, map_size=map_size)

    current_size = 0
    progress = 0
    TIMER.start(f"Writing data")
    
    with env.begin(write=True) as txn:
        for board_tensor, metadata, enpassant, halfmoves, policy, value, legal_mask in encode_pgn_file(pgn_path, history_length, num_games, chunk_size):  # type: ignore
            if shuffle:
                indices = np.random.permutation(board_tensor.shape[0])

                board_tensor = board_tensor[indices]
                metadata = metadata[indices]
                enpassant = enpassant[indices]
                halfmoves = halfmoves[indices]
                policy = policy[indices]
                value = value[indices]
                legal_mask = legal_mask[indices]

            for i in range(board_tensor.shape[0]):
                if current_size >= max_samples:
                    break

                # Serialize sample as a tuple
                sample = (board_tensor[i], metadata[i], enpassant[i], halfmoves[i], policy[i], value[i], legal_mask[i])
                key = f"{current_size:08}".encode("ascii")
                txn.put(key, pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))

                current_size += 1

            while current_size / max_samples > progress:
                TIMER.lap("Writing data", current_size, max_samples)
                print(f"Currently: {current_size}/{max_samples}")
                progress += 0.002

            if current_size >= max_samples:
                break

        # Store metadata as a special key
        txn.put(b"__len__", pickle.dumps(current_size))
        txn.put(b"__history_length__", pickle.dumps(history_length))

    env.close()
    TIMER.stop("Writing data")
    print(f"Finished writing {current_size} samples to LMDB")



if __name__ == "__main__":
    random.seed(2025)

    train_samples = 20_000_000
    val_samples = 500_000

    data_path = r'/teamspace/studios/this_studio/chess_bot/datasets/raw/CCRL-4040/CCRL-4040.[2173847].pgn'
    # h5py_path = r'/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040.h5'
    lmdb_path_train = f'/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040-train-{train_samples}-{val_samples}-{OPENING_SAMPLE_PROB}-{MIDGAME_SAMPLE_PROB}-{ENDGAME_SAMPLE_PROB}.lmdb'
    lmdb_path_val = f'/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040-val-{train_samples}-{val_samples}-{OPENING_SAMPLE_PROB}-{MIDGAME_SAMPLE_PROB}-{ENDGAME_SAMPLE_PROB}.lmdb'

    if os.path.exists(lmdb_path_train) or os.path.exists(lmdb_path_val):
        raise FileExistsError(f"LMDB files already exist at {lmdb_path_train} or {lmdb_path_val}")

    store_lmdb(pgn_path=data_path, lmdb_path=lmdb_path_train, max_samples=train_samples, history_length=1, chunk_size=20, shuffle=False)
    store_lmdb(pgn_path=data_path, lmdb_path=lmdb_path_val, max_samples=val_samples, history_length=1, chunk_size=20, shuffle=False)

    # TIMER.start("Loading data batch")
    # for board_tensor, metadata, enpassant, halfmoves, policy, value, legal_mask in encode_pgn_file(data_path, history_length=HISTORY_LEN, num_games=1000, chunk_size=50):
    #     TIMER.lap("Loading data batch", 1, 1)
    #     print(board_tensor.shape, metadata.shape, enpassant.shape, halfmoves.shape, policy.shape, value.shape, legal_mask.shape)
