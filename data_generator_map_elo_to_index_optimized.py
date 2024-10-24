import random
import time
import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from stockfish import Stockfish
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChessSimulator:
    def __init__(self, positions_file='board_positions.txt'):
        logging.info("Initializing ChessSimulator...")
        self.board_positions = self.read_positions_from_file(positions_file)

    def read_positions_from_file(self, filename: str) -> List[str]:
        logging.info(f"Reading positions from file: {filename}...")
        with open(filename, 'r') as f:
            board_positions = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(board_positions)} positions from {filename}.")
        return board_positions

    def generate_board_state(self, i) -> str:
        return self.board_positions[i]

def create_stockfish_instance(stockfish_path=None):
    try:
        stockfish = Stockfish(path=stockfish_path) if stockfish_path else Stockfish()
        stockfish.update_engine_parameters({
            "MultiPV": 10,  # Get top 10 moves
            "Threads": 8    # Set Stockfish to use 8 threads
        })
        return stockfish
    except Exception as e:
        logging.error(f"Error creating Stockfish instance: {str(e)}")
        raise

def get_move_by_elo(stockfish: Stockfish, fen: str, elo: int) -> str:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            stockfish.set_fen_position(fen)
            top_moves = stockfish.get_top_moves(10)
            
            if not top_moves:
                return None

            max_index = len(top_moves) - 1
            index = min(int((3200 - elo) / 300), max_index)
            index = min(max(0, int(np.random.normal(index, 1))), max_index)
            
            chosen_move = top_moves[index]['Move']
            return chosen_move
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                logging.error(f"Error in get_move_by_elo: {str(e)}")
                return None

def get_move_score(stockfish: Stockfish, fen: str, move: str, opponent_is_black: bool) -> int:
    try:
        stockfish.set_fen_position(fen)
        if move:
            stockfish.make_moves_from_current_position([move])
        evaluation = stockfish.get_evaluation()
        
        if evaluation['type'] == 'cp':
            return evaluation['value']
        elif evaluation['type'] == 'mate':
            return 10000 if (evaluation['value'] > 0) != opponent_is_black else -10000
    except Exception as e:
        logging.error(f"Error in get_move_score: {str(e)}")
        return 0

def get_cpl(stockfish: Stockfish, fen: str, move: str) -> float:
    opponent_is_black = " b" in fen
    best_move = get_move_by_elo(stockfish, fen, 3200)
    
    if not best_move:
        return None
    
    best_move_score = get_move_score(stockfish, fen, best_move, opponent_is_black)
    actual_move_score = get_move_score(stockfish, fen, move, opponent_is_black)
    
    cpl = abs(best_move_score - actual_move_score)
    return cpl

def simulate_moves(stockfish, task):
    fen, elo, is_cheating, num_trials = task
    cpls = []
    for _ in range(num_trials):
        move = get_move_by_elo(stockfish, fen, 3200 if is_cheating else elo)
        if not move:
            continue
        cpl = get_cpl(stockfish, fen, move)
        if cpl is not None:
            cpls.append(cpl)
    avg_cpl = np.mean(cpls) if cpls else None
    return (avg_cpl, elo, 1 if is_cheating else 0, fen)

def normalize_cpl_per_position(df: pd.DataFrame) -> pd.DataFrame:
    df['CPL_normalized'] = df.groupby('FEN')['CPL'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
    )
    return df

def generate_normalized_dataset(
    num_board_positions: int = 1000,
    num_trials: int = 5,
    elo_range: Tuple[int, int, int] = (500, 3200, 100),
    positions_file: str = 'board_positions.txt',
    stockfish_path: str = None,
    output_file: str = 'chess_dataset_normalized.csv'
) -> None:
    logging.info(f"Starting dataset generation with {num_board_positions} positions and {num_trials} trials per position.")
    start_time = time.time()

    simulator = ChessSimulator(positions_file=positions_file)
    board_positions = [simulator.generate_board_state(i) for i in range(num_board_positions)]
    elos = list(range(elo_range[0], elo_range[1] + 1, elo_range[2]))

    tasks = [(fen, elo, is_cheating, num_trials)
             for fen in board_positions
             for elo in elos
             for is_cheating in [True, False]]

    stockfish = create_stockfish_instance(stockfish_path)

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:  # Use 8 threads for parallel processing
        futures = [executor.submit(simulate_moves, stockfish, task) for task in tasks]
        for future in tqdm(futures, desc="Generating Dataset"):
            results.append(future.result())

    df = pd.DataFrame(results, columns=['CPL', 'ELO', 'Cheat', 'FEN'])
    df = normalize_cpl_per_position(df)

    df.to_csv(output_file, index=False)
    logging.info(f"Dataset saved to {output_file}")

    end_time = time.time()
    logging.info(f"Dataset generation completed in {end_time - start_time:.2f} seconds.")

def plot_feature_distributions(dataset_file: str = 'chess_dataset.csv') -> None:
    logging.info(f"Loading dataset from {dataset_file} for plotting.")
    df = pd.read_csv(dataset_file)

    cheat_df = df[df['Cheat'] == 1]
    non_cheat_df = df[df['Cheat'] == 0]

    plt.figure(figsize=(12, 6))
    plt.hist(non_cheat_df['CPL'], bins=50, alpha=0.5, label='Non-Cheating', color='blue')
    plt.hist(cheat_df['CPL'], bins=50, alpha=0.5, label='Cheating', color='red')
    plt.xlabel('Centipawn Loss (CPL)')
    plt.ylabel('Frequency')
    plt.title('CPL Distribution for Cheating vs Non-Cheating Players')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.scatter(non_cheat_df['ELO'], non_cheat_df['CPL'], alpha=0.5, label='Non-Cheating', color='blue')
    plt.scatter(cheat_df['ELO'], cheat_df['CPL'], alpha=0.5, label='Cheating', color='red')
    plt.xlabel('ELO Rating')
    plt.ylabel('Centipawn Loss (CPL)')
    plt.title('CPL vs ELO for Cheating and Non-Cheating Players')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    generate_normalized_dataset(
        num_board_positions=10000,
        num_trials=10,
        elo_range=(500, 3200, 100),
        positions_file='board_positions_large.txt',
        stockfish_path=None,  # Replace with actual path
        output_file='chess_dataset_large.csv'
    )

    plot_feature_distributions('chess_dataset_large.csv')
