import random
from stockfish import Stockfish
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

class ChessSimulator:
    def __init__(self, positions_file='board_positions.txt'):
        self.stockfish = self.create_stockfish_instance()
        self.cpl_values: List[float] = []
        self.cpl_rolling_average: float = 0
        self.random_positions = self.read_positions_from_file(positions_file)

    def create_stockfish_instance(self):
        return Stockfish()

    def reboot_stockfish(self):
        print("Rebooting Stockfish...")
        self.stockfish = self.create_stockfish_instance()
        time.sleep(1)  # Give some time for the engine to initialize

    def read_positions_from_file(self, filename):
        with open(filename, 'r') as f:
            return [line.strip() for line in f]

    def generate_random_board_state(self) -> str:
        return random.choice(self.random_positions)

    def update_cpl_average(self, new_cpl: float) -> None:
        self.cpl_values.append(new_cpl)
        if len(self.cpl_values) > 10:
            self.cpl_values.pop(0)
        self.cpl_rolling_average = sum(self.cpl_values) / len(self.cpl_values)

    def get_best_move(self, fen: str, elo: int) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.stockfish.set_fen_position(fen)
                self.stockfish.set_elo_rating(elo)
                best_move = self.stockfish.get_best_move()
                return best_move if best_move else None
            except Exception as e:
                print(f"Stockfish error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    self.reboot_stockfish()
                else:
                    print("Max retries reached. Skipping this move.")
                    return None

    def get_move_score(self, fen: str, move: str) -> int:
        try:
            self.stockfish.set_fen_position(fen)
            if move:
                self.stockfish.make_moves_from_current_position([move])
            eval = self.stockfish.get_evaluation()
            return eval['value'] if eval['type'] == 'cp' else (10000 if eval['value'] > 0 else -10000)
        except Exception as e:
            print(f"Error in get_move_score: {str(e)}")
            self.reboot_stockfish()
            return 0  # Return a neutral score in case of error

    def get_cpl(self, fen: str, move: str, elo: int) -> float:
        best_move = self.get_best_move(fen, 3200)  # Assuming 3200 is the highest ELO
        if not best_move:
            return 0  # Return 0 CPL if no legal move is available
        best_move_score = self.get_move_score(fen, best_move)
        actual_move_score = self.get_move_score(fen, move)
        return abs(best_move_score - actual_move_score)

    def simulate_move(self, fen: str, elo: int, is_cheating: bool) -> Tuple[str, float]:
        if is_cheating:
            move = self.get_best_move(fen, 3200)
        else:
            move = self.get_best_move(fen, elo)

        if not move:
            return None, 0  # Return None and 0 CPL if no legal move is available

        cpl = self.get_cpl(fen, move, elo)
        self.update_cpl_average(cpl)
        return move, cpl

def run_experiment(num_board_positions: int, num_moves: int, elo_range: Tuple[int, int, int], cpl_thresholds: Dict[Tuple[int, int], float]) -> Dict[int, Tuple[float, float]]:
    simulator = ChessSimulator()
    results = {}

    # Generate a fixed set of board positions to use for all ELO ratings
    board_positions = [simulator.generate_random_board_state() for _ in range(num_board_positions)]

    total_iterations = len(range(elo_range[0], elo_range[1] + 1, elo_range[2]))
    
    with tqdm(total=total_iterations, desc="Overall Progress") as pbar_overall:
        for elo in range(elo_range[0], elo_range[1] + 1, elo_range[2]):
            correct_detections = 0
            cpl_sum_non_cheating = 0
            non_cheating_count = 0

            with tqdm(total=num_board_positions, desc=f"Processing board positions (ELO: {elo})", leave=False) as pbar_positions:
                for fen in board_positions:
                    simulator.cpl_values = []
                    simulator.cpl_rolling_average = 0
                    is_cheating = random.choice([True, False])

                    trial_cpl_sum = 0
                    moves_made = 0
                    with tqdm(total=num_moves, desc=f"Moves for position: {fen[:20]}...", leave=False) as pbar_num_moves:
                        for _ in range(num_moves):
                            try:
                                move, cpl = simulator.simulate_move(fen, elo, is_cheating)
                                if move is None:
                                    break  # No legal moves available, end this trial
                                
                                trial_cpl_sum += cpl
                                moves_made += 1
                                
                                fen = simulator.stockfish.get_fen_position()
                                pbar_num_moves.update(1)
                            except Exception as e:
                                print(f"Error during move simulation: {str(e)}")
                                simulator.reboot_stockfish()

                    if moves_made > 0:
                        # Check if the CPL rolling average is below the threshold for this ELO range
                        detected_cheating = False
                        for (elo_min, elo_max), threshold in cpl_thresholds.items():
                            if elo_min <= elo <= elo_max and simulator.cpl_rolling_average < threshold:
                                detected_cheating = True
                                break

                        if is_cheating == detected_cheating:
                            correct_detections += 1

                        if not is_cheating:
                            cpl_sum_non_cheating += trial_cpl_sum / moves_made
                            non_cheating_count += 1

                    pbar_positions.update(1)

            accuracy = correct_detections / num_board_positions
            avg_cpl = cpl_sum_non_cheating / non_cheating_count if non_cheating_count > 0 else 0
            results[elo] = (accuracy, avg_cpl)
            pbar_overall.update(1)

    return results

def plot_results(results: Dict[int, Tuple[float, float]]) -> None:
    elos = list(results.keys())
    accuracies, cpl_averages = zip(*results.values())

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(elos, accuracies, 'ro-')
    plt.xlabel('ELO Rating')
    plt.ylabel('Accuracy')
    plt.title('Cheat Detection Accuracy vs ELO Rating')
    plt.ylim(0, 1)  # Set y-axis limits for accuracy
    plt.grid(True)
    plt.show()

    # Plot average CPL
    plt.figure(figsize=(10, 6))
    plt.plot(elos, cpl_averages, 'bo-')
    plt.xlabel('ELO Rating')
    plt.ylabel('Average Centipawn Loss (Non-Cheating)')
    plt.title('Average Centipawn Loss vs ELO Rating (Non-Cheating Players)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Define initial CPL thresholds for different ELO ranges
    cpl_thresholds = {
        (0, 799): 150,
        (800, 1400): 100,
        (1401, 2000): 15,
        (2001, 2400): 10,
        (2401, 3200): 5
    }

    results = run_experiment(num_board_positions=10, num_moves=10, elo_range=(500, 2800, 100), cpl_thresholds=cpl_thresholds)
    plot_results(results)