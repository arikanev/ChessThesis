import random
import time
import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from stockfish import Stockfish
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChessSimulator:
    def __init__(self, positions_file='board_positions.txt', stockfish_path=None):
        logging.info("Initializing ChessSimulator...")
        self.stockfish = self.create_stockfish_instance(stockfish_path)
        self.random_positions = self.read_positions_from_file(positions_file)

    def create_stockfish_instance(self, stockfish_path):
        logging.info(f"Creating Stockfish instance (path: {stockfish_path})...")
        stockfish = Stockfish(path=stockfish_path) if stockfish_path else Stockfish()
        stockfish.update_engine_parameters({"MultiPV": 10})  # Get top 10 moves
        return stockfish

    def reboot_stockfish(self):
        logging.warning("Rebooting Stockfish due to an error...")
        self.stockfish = self.create_stockfish_instance(None)
        time.sleep(1)

    def read_positions_from_file(self, filename: str) -> List[str]:
        logging.info(f"Reading positions from file: {filename}...")
        with open(filename, 'r') as f:
            positions = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(positions)} positions from {filename}.")
        return positions

    def generate_random_board_state(self) -> str:
        return random.choice(self.random_positions)

    def get_move_by_elo(self, fen: str, elo: int) -> str:
        logging.debug(f"Getting move for FEN: {fen[:20]}... at ELO: {elo}")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.stockfish.set_fen_position(fen)
                top_moves = self.stockfish.get_top_moves(10)
                
                if not top_moves:
                    logging.warning("No legal moves found.")
                    return None

                # Map ELO to move index
                max_index = len(top_moves) - 1
                index = min(int((3200 - elo) / 300), max_index)
                
                # Add some randomness
                index = min(max(0, int(np.random.normal(index, 1))), max_index)
                
                chosen_move = top_moves[index]['Move']
                return chosen_move
            except Exception as e:
                logging.error(f"Stockfish error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    self.reboot_stockfish()
                else:
                    logging.error("Max retries reached. Skipping this move.")
                    return None

    def get_move_score(self, fen: str, move: str) -> int:
        try:
            self.stockfish.set_fen_position(fen)
            if move:
                self.stockfish.make_moves_from_current_position([move])
            eval = self.stockfish.get_evaluation()
            return eval['value'] if eval['type'] == 'cp' else (10000 if eval['value'] > 0 else -10000)
        except Exception as e:
            logging.error(f"Error in get_move_score: {str(e)}")
            self.reboot_stockfish()
            return 0

    def get_cpl(self, fen: str, move: str) -> float:
        logging.debug(f"Calculating CPL for move: {move}")
        best_move = self.get_move_by_elo(fen, 3200)
        print("best move: ", best_move)
        print("actual move: ", move)
        if not best_move:
            return None
        best_move_score = self.get_move_score(fen, best_move)
        actual_move_score = self.get_move_score(fen, move)
        cpl = abs(best_move_score - actual_move_score)
        print("CPL: ", cpl)
        return cpl

    def simulate_move(self, fen: str, elo: int, is_cheating: bool, num_trials: int) -> float:
        logging.info(f"Simulating {num_trials} trials for FEN: {fen[:20]}... at ELO: {elo}, cheating: {is_cheating}")
        cpls = []
        for _ in range(num_trials):
            if is_cheating:
                move = self.get_move_by_elo(fen, 3200)
            else:
                move = self.get_move_by_elo(fen, elo)

            if not move:
                logging.warning("No legal move found, skipping this position.")
                continue

            cpl = self.get_cpl(fen, move)
            if cpl is not None:
                cpls.append(cpl)

        if cpls:
            avg_cpl = np.mean(cpls)
            print("AVG CPL: ", avg_cpl)
            return avg_cpl
        else:
            return None

def generate_dataset(
    num_board_positions: int = 1000,
    num_trials: int = 5,
    elo_range: Tuple[int, int, int] = (500, 3200, 100),
    positions_file: str = 'board_positions.txt',
    stockfish_path: str = None,
    output_file: str = 'chess_dataset.csv'
) -> None:
    logging.info(f"Starting dataset generation with {num_board_positions} positions and {num_trials} trials per position.")
    start_time = time.time()

    simulator = ChessSimulator(positions_file=positions_file, stockfish_path=stockfish_path)
    X_data = []
    y_labels = []

    board_positions = [simulator.generate_random_board_state() for _ in range(num_board_positions)]
    logging.info(f"Generated {len(board_positions)} random board positions.")

    elos = list(range(elo_range[0], elo_range[1] + 1, elo_range[2]))
    total_iterations = len(elos) * num_board_positions * 2  # *2 for cheating and non-cheating trials

    with tqdm(total=total_iterations, desc="Generating Dataset") as pbar_overall:
        for elo in elos:
            for fen in board_positions:
                for is_cheating in [True, False]:
                    logging.debug(f"Simulating move for ELO {elo}, FEN: {fen[:20]}...")

                    avg_cpl = simulator.simulate_move(fen, elo, is_cheating, num_trials)
                    if avg_cpl is None:
                        logging.warning("No valid CPL found, skipping to next position.")
                        continue

                    X_data.append([avg_cpl, elo])
                    y_labels.append(1 if is_cheating else 0)

                    pbar_overall.update(1)

    df = pd.DataFrame(X_data, columns=['CPL', 'ELO'])
    df['Cheat'] = y_labels

    df = df.sample(frac=1).reset_index(drop=True)

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
    generate_dataset(
        num_board_positions=1,   # Adjust based on desired dataset size
        num_trials=10,
        elo_range=(500, 3200, 100), # ELO ratings from 500 to 3200 in steps of 100
        positions_file='board_positions_old.txt', # Ensure this file exists with valid FENs
        stockfish_path=None,        # Specify path if Stockfish is not in PATH
        output_file='chess_dataset.csv' # Output CSV file
    )

    plot_feature_distributions('chess_dataset.csv')