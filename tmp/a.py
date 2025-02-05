import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt

curdir = Path(__file__).parent.resolve()

df = pd.read_csv(f"{curdir}/a.csv")

# combinations = []
# num_combs = 3
# tc = len(df.columns)
# total_combinations = math.factorial(tc)/(math.factorial(num_combs)*math.factorial(tc-num_combs))
# for cols in tqdm(itertools.combinations(df.columns, num_combs), total=total_combinations):
#     mask = True
#     for c in cols:
#         mask &= df[c] > 0
#     weight = int(mask.sum())
#     if weight > 0:
#         combinations += [(weight, cols)]
# combinations = sorted(combinations, key=lambda x: x[0])
# print(combinations)

total_cols = len(df.columns)
adj_mat = np.zeros((total_cols, total_cols))
for i, node_a in enumerate(df.columns):
    for j in range(i + 1, total_cols):
        node_b = df.columns[j]
        weight = ((df[node_a] != 0) & (df[node_b] != 0)).sum()
        adj_mat[i, j] = weight


print(adj_mat)

min_edge_weight = 3
G = nx.from_numpy_array(adj_mat, nodelist=df.columns)


def filter_edge(n1, n2):
    return G[n1][n2]["weight"] > min_edge_weight


subG = nx.subgraph_view(G, filter_edge=filter_edge)

remaining_nodes = set()
for n1, n2 in subG.edges:
    remaining_nodes.add(n1)
    remaining_nodes.add(n2)
subG = nx.subgraph_view(subG, filter_node=lambda x: x in remaining_nodes)
plt.figure(figsize=(20, 20))

colors = [
    "#f44336",
    "#e81e63",
    "#9c27b0",
    "#673ab7",
    "#3f51b5",
    "#2196f3",
    "#03a9f4",
    "#00bcd4",
    "#009688",
    "#4caf50",
    "#8bc34a",
    "#cddc39",
    "#c71585",
    "#00fa9a",
    "#ff9800",
    "#ff5722",
    "#2f4f4f",
    "#800000",
]
color_map = dict([(n, colors[i]) for (i, n) in enumerate(subG.nodes)])
layout = nx.circular_layout(subG)
for edge in subG.edges(data="weight"):
    start_node = edge[0]
    node_color = color_map[start_node]
    nx.draw_networkx_edges(
        subG,
        layout,
        edgelist=[edge],
        width=(edge[2] / 3),
        alpha=(edge[2] / 10),
        edge_color=node_color,
    )
nx.draw_networkx_nodes(subG, layout, node_color=colors)
scale = 1.1
label_pos = {node: (scale * x, scale * y) for node, (x, y) in layout.items()}
nx.draw_networkx_labels(subG, label_pos, font_size=17, font_color=color_map)
# nx.draw_networkx_labels(subG, layout, font_size=20)

# Extract x and y coordinates of nodes
x_coords = [x for x, y in layout.values()]
y_coords = [y for x, y in layout.values()]
padding = 0.2  # Adjust padding as needed
plt.xlim(min(x_coords) - padding, max(x_coords) + padding)
plt.ylim(min(y_coords) - padding, max(y_coords) + padding)
# plt.axis("off")

plt.savefig(f"{curdir}/a.png", bbox_inches="tight", pad_inches=0)

print(sorted(G.edges(data="weight"), key=lambda x: x[2]))
