from os import name
import torch
import torch.optim as optim
import chess
import numpy as np
from collections import defaultdict, deque
from typing import Tuple, List, Dict
import copy

from data_preprocess import board_to_planes, encode_board, move_to_index
from model import ChessViT
from run_timer import TIMER


class MCTSNode:
    """Node in the MCTS tree."""
    def __init__(self, board: chess.Board, history: deque, parent=None, prior: float = 0.0):
        self.board = board
        self.history = history
        self.parent = parent
        self.prior = prior
        self.children = {}  # Move -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        
    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def is_expanded(self) -> bool:
        return len(self.children) > 0
    
    def select_child(self, c_puct: float = 1.5) -> Tuple[chess.Move, 'MCTSNode']:
        """Select child using UCB formula."""
        best_score = -float('inf')
        best_move = None
        best_child = None
        
        for move, child in self.children.items():
            # UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            q_value = child.value()
            u_value = c_puct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        
        return best_move, best_child # type: ignore


def policy_to_move_probs(board: chess.Board, policy: torch.Tensor) -> Dict[chess.Move, float]:
    """Convert policy vector to move probabilities for legal moves."""
    legal_moves = list(board.legal_moves)
    move_probs = {}
    
    for move in legal_moves:
        idx = move_to_index(move, board)
        move_probs[move] = policy[idx].item()
    
    # Normalize
    total = sum(move_probs.values())
    if total > 0:
        move_probs = {m: p / total for m, p in move_probs.items()}
    else:
        # Uniform if all zero
        uniform_prob = 1.0 / len(legal_moves)
        move_probs = {m: uniform_prob for m in legal_moves}
    
    return move_probs


def mcts_search(board: chess.Board, history: deque, model, num_simulations: int = 800,
                c_puct: float = 1.5, add_noise: bool = True) -> Dict[chess.Move, float]:
    """
    Perform MCTS search and return visit count distribution over moves.
    
    Args:
        board: Current chess position
        model: Neural network model
        num_simulations: Number of MCTS simulations
        c_puct: Exploration constant
        add_noise: Whether to add Dirichlet noise at root
    
    Returns:
        Dictionary mapping moves to visit proportions
    """

    root = MCTSNode(board.copy(), history=history)
    
    # Add Dirichlet noise to root for exploration
    if add_noise:
        noise = np.random.dirichlet([0.3] * len(list(board.legal_moves)))
        noise_weight = 0.25
    
    for _ in range(num_simulations):
        node = root
        search_path = [node]
        current_board = board.copy()
        current_history = history.copy()
        
        # Selection: traverse tree until we reach a leaf
        while node.is_expanded() and not current_board.is_game_over():
            move, node = node.select_child(c_puct)
            current_history.append(board_to_planes(current_board))
            current_board.push(move)
            search_path.append(node)
        
        # Check if game is over
        if current_board.is_game_over():
            result = current_board.result()
            if result == "1-0":
                value = 1.0 if current_board.turn == chess.BLACK else -1.0
            elif result == "0-1":
                value = -1.0 if current_board.turn == chess.BLACK else 1.0
            else:
                value = 0.0
        else:
            # Expansion and evaluation
            with torch.no_grad():
                board_tensor = torch.as_tensor(encode_board(current_board, current_history)).unsqueeze(0)
                model_policy, model_value = model(board_tensor)
                value = model_value.item()
                policy = torch.softmax(model_policy.squeeze(0), dim=0)
            
            # Expand node
            move_probs = policy_to_move_probs(current_board, policy)
            
            for i, (move, prob) in enumerate(move_probs.items()):
                child_board = current_board.copy()
                child_history = current_history.copy()
                child_board.push(move)
                
                prior = prob
                # Add Dirichlet noise at root
                if node == root and add_noise:
                    prior = (1 - noise_weight) * prior + noise_weight * noise[i]
                
                node.children[move] = MCTSNode(child_board, history=child_history, parent=node, prior=prior)
        
        # Backpropagation
        for node in reversed(search_path):
            node.value_sum += value if current_board.turn == chess.WHITE else -value
            node.visit_count += 1
            value = -value  # Flip for opponent
    
    # Return visit count distribution
    visit_counts = {move: child.visit_count for move, child in root.children.items()}
    total = sum(visit_counts.values())
    return {move: count / total for move, count in visit_counts.items()}


def generate_self_play_game(model, history_length=8, temperature: float = 1.0,
                           num_simulations: int = 800) -> List[Tuple]:
    """
    Generate a single self-play game.
    
    Args:
        model: Neural network model
        temperature: Temperature for move selection (higher = more exploration)
        num_simulations: MCTS simulations per move
    
    Returns:
        List of (board_tensor, policy_target, value_target) tuples
    """
    TIMER.start("Generating self-play game")
    board = chess.Board()
    history = deque(maxlen=history_length)
    examples = []
    move_count = 0

    # Fill initial history with empty boards
    empty_planes = np.zeros((12, 8, 8), dtype=np.float32)
    for _ in range(history_length):
        history.append(empty_planes)
    
    while not board.is_game_over():
        # Use temperature for first 30 moves, then greedy
        temp = temperature if move_count < 30 else 0.1
        
        # Run MCTS
        move_probs = mcts_search(
            board=board,
            history=history,
            model=model, 
            num_simulations=num_simulations, add_noise=True
        )
        
        # Store training example (we'll assign value at end)
        board_tensor = encode_board(board, history)
        
        # Create policy target
        policy_target = torch.zeros(4672)
        for move, prob in move_probs.items():
            idx = move_to_index(move, board)
            policy_target[idx] = prob
        
        examples.append((board_tensor, policy_target, board.turn))
        
        # Sample move based on visit counts with temperature
        moves = list(move_probs.keys())
        probs = list(move_probs.values())
        
        if temp < 0.2:
            # Greedy
            move = moves[np.argmax(probs)]
        else:
            # Sample with temperature
            probs = np.array(probs) ** (1.0 / temp)
            probs = probs / probs.sum()
            move = np.random.choice(moves, p=probs) # type: ignore
        
        board.push(move)
        move_count += 1

        TIMER.lap("Generating self-play game", move_count, move_count)

    TIMER.stop("Generating self-play game")
    TIMER.start("Processing generaged game data")
    
    # Assign game outcome to all positions
    result = board.result()
    if result == "1-0":
        game_value = 1.0
    elif result == "0-1":
        game_value = -1.0
    else:
        game_value = 0.0
    
    # Create final training examples with correct values
    training_examples = []
    for board_tensor, policy_target, turn in examples:
        # Value is from perspective of player to move
        value = game_value if turn == chess.WHITE else -game_value
        training_examples.append((board_tensor, policy_target, value))

    TIMER.stop("Processing generaged game data")
    
    return training_examples


def train_from_self_play(model, optimizer, num_games: int = 100,
                        num_simulations: int = 800, batch_size: int = 256,
                        epochs_per_iteration: int = 10, device='cuda'):
    """
    Generate self-play games and train the model.
    
    Args:
        model: Neural network model
        optimizer: PyTorch optimizer
        num_games: Number of self-play games to generate
        num_simulations: MCTS simulations per move
        batch_size: Training batch size
        epochs_per_iteration: Number of epochs to train on collected data
        device: Device to train on
    """
    TIMER.start("Generating MCTS examples")
    
    model.eval()
    
    # Generate self-play games
    print(f"Generating {num_games} self-play games...")
    all_examples = []
    
    for game_idx in range(num_games):
        examples = generate_self_play_game(model, history_length=1,
                                          temperature=1.0, num_simulations=num_simulations)
        all_examples.extend(examples)
        
        if (game_idx + 1) % 10 == 0:
            print(f"Generated {game_idx + 1}/{num_games} games, {len(all_examples)} positions")
            TIMER.lap("Generating MCTS examples", game_idx + 1, num_games)
    
    TIMER.stop("Generating MCTS examples")
    print(f"Total training examples: {len(all_examples)}")
    TIMER.start(f"Training for {epochs_per_iteration} epochs")

    # Train on collected data
    model.train()
    
    for epoch in range(epochs_per_iteration):
        np.random.shuffle(all_examples)
        total_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(all_examples), batch_size):
            batch = all_examples[i:i+batch_size]
            
            # Prepare batch
            boards = torch.stack([ex[0] for ex in batch]).to(device)
            policy_targets = torch.stack([ex[1] for ex in batch]).to(device)
            value_targets = torch.tensor([ex[2] for ex in batch], dtype=torch.float32).to(device)
            
            # Forward pass
            pred_values, pred_policies = model(boards)
            pred_values = pred_values.squeeze()
            
            # Loss computation
            value_loss = torch.nn.functional.mse_loss(pred_values, value_targets)
            policy_loss = -torch.sum(policy_targets * torch.log_softmax(pred_policies, dim=1)) / len(batch)
            
            loss = value_loss + policy_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        
        TIMER.lap(f"Training for {epochs_per_iteration} epochs", epoch + 1, epochs_per_iteration)
        print(f"Epoch {epoch+1}/{epochs_per_iteration}: "
              f"Loss={avg_loss:.4f}, Value={avg_value_loss:.4f}, Policy={avg_policy_loss:.4f}")
    
    TIMER.stop(f"Training for {epochs_per_iteration} epochs")
    print("Training iteration complete!")


# Example usage:
"""
# Assuming you have:
# - model: Your PyTorch model with forward(x) -> (value, policy)
# - board_to_tensor: Function that converts chess.Board to 18x8x8 tensor
# - optimizer: Your optimizer (e.g., Adam)

train_from_self_play(
    model=model,
    optimizer=optimizer,
    num_games=100,
    num_simulations=800,
    batch_size=256,
    epochs_per_iteration=10,
    device='cuda'
)
"""

if __name__ == "__main__":

    TIMER.start("Initializing")

    checkpoint_dir = r"/teamspace/studios/this_studio/chess_bot/results/checkpoints/dataset-2m_lr-1e-4/model_epoch_78.pt"

    lr = 1e-4
    weight_decay = 0.05

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # device = "cpu"

    model = ChessViT()
    model.load_state_dict(torch.load(checkpoint_dir, map_location=torch.device(device)))
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    TIMER.stop("Initializing")

    train_from_self_play(
        model=model,
        optimizer=optimizer, 
        num_games=100,
        num_simulations=800,
        batch_size=256,
        epochs_per_iteration=10,
        device=device
    )