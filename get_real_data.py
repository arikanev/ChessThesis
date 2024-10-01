import chess
import chess.pgn
import pandas as pd
from stockfish import Stockfish
from tqdm import tqdm
import io

def calculate_cpl(stockfish, fen, actual_move):
    stockfish.set_skill_level(20)
    stockfish.set_fen_position(fen)
    best_move = stockfish.get_best_move()
    stockfish.make_moves_from_current_position([best_move])
    best_score = stockfish.get_evaluation()['value']
    stockfish.set_fen_position(fen)
    stockfish.make_moves_from_current_position([actual_move])
    actual_score = stockfish.get_evaluation()['value']
    cpl = max(0, abs(best_score - actual_score))
    return cpl

def process_games(games_data, stockfish, num_games=10):
    data = []
    game_count = 0  # Track number of games processed
    for _, row in tqdm(games_data.iterrows(), total=min(num_games, len(games_data))):
        pgn = io.StringIO(row['Game'])
        game = chess.pgn.read_game(pgn)
        if game is None:
            continue
        
        elo_white = row['Elo White']
        elo_black = row['Elo Black']
        cheat_white_list = [int(x) for x in row['Liste cheat white']]
        cheat_black_list = [int(x) for x in row['Liste cheat black']]
        
        board = game.board()
        move_count = 0
        for move in game.mainline_moves():
            fen = board.fen()
            is_white_turn = board.turn == chess.WHITE
            player_elo = elo_white if is_white_turn else elo_black
            
            # Use the correct index for the cheating list
            if is_white_turn:
                is_cheating = cheat_white_list[move_count // 2]  # Integer division by 2 because each player makes a move
            else:
                is_cheating = cheat_black_list[move_count // 2]
            
            cpl = calculate_cpl(stockfish, fen, move.uci())
            print("FEN: ", fen, "Is_Cheating: ", is_cheating, "CPL: ", cpl)
            
            data.append({
                'FEN': fen,
                'Move': move.uci(),
                'ELO': player_elo,
                'CPL': cpl,
                'Is_Cheating': is_cheating
            })
            
            board.push(move)
            move_count += 1
        
        game_count += 1  # Increment after processing a full game
        if game_count >= num_games:
            break
    
    return pd.DataFrame(data)


# Initialize Stockfish with maximum skill level
stockfish = Stockfish(depth=20)
stockfish.set_skill_level(20)

# Load the CSV data
games_data = pd.read_csv('Games.csv')

# Process games and calculate CPL
df = process_games(games_data, stockfish, num_games=10)

# Group by ELO ranges and calculate average CPL
df['ELO_Range'] = pd.cut(df['ELO'].astype(float),
                         bins=[-1, 1000, 1500, 2000, 2500, 3000, float('inf')],
                         labels=['<1000', '1001-1500', '1501-2000', '2001-2500', '2501-3000', '3000+'])

cpl_by_elo = df.groupby(['ELO_Range', 'Is_Cheating'])['CPL'].mean().unstack()
print("Average CPL by ELO range and cheating status:")
print(cpl_by_elo)

# Calculate cheating percentage by ELO range
cheating_percentage = df.groupby('ELO_Range')['Is_Cheating'].mean() * 100
print("\nCheating percentage by ELO range:")
print(cheating_percentage)

# Save results
df.to_csv('chess_cpl_data.csv', index=False)
cpl_by_elo.to_csv('average_cpl_by_elo_and_cheating.csv')
cheating_percentage.to_csv('cheating_percentage_by_elo.csv')

print("\nResults saved to CSV files.")