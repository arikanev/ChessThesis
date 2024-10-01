import matplotlib.pyplot as plt
import numpy as np

def get_index(elo, max_index=20):
    return min(int((3200 - elo) / 300), max_index)

elo_range = np.arange(500, 3201, 50)
indices = [get_index(elo) for elo in elo_range]

plt.figure(figsize=(10, 6))
plt.plot(elo_range, indices, 'b-')
plt.xlabel('ELO Rating')
plt.ylabel('Move Rank Index')
plt.title('Relationship between ELO Rating and Move Rank Selection')
plt.grid(True)
plt.ylim(0, max(indices) + 1)
plt.savefig('elo_move_rank_plot.png', dpi=300, bbox_inches='tight')
plt.close()