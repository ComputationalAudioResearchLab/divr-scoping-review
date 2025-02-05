import pandas as pd
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

curdir = Path(__file__).parent.resolve()
df = pd.read_csv(f"{curdir}/a.csv")


G = nx.DiGraph()

for idx, row in df.iterrows():
    row = row[row > 0].sort_values(ascending=True)
    nodes = row.keys()
    if len(nodes) > 0:
        first_node = nodes[0]
        if not G.has_node(first_node):
            G.add_node(first_node)
        prev_node = first_node
        for node in nodes[1:]:
            if not G.has_node(node):
                G.add_node(node)
            G.add_edge(prev_node, node)
            prev_node = node

plt.figure(figsize=(20, 20))

# # layout = nx.circular_layout(G)
# layout = nx.spiral_layout(G)
# for edge in G.edges(data="weight"):
#     # nx.draw_networkx_edges(G, layout, edgelist=[edge], alpha=(edge[2]/20))
#     nx.draw_networkx_edges(G, layout, edgelist=[edge])
# nx.draw_networkx_nodes(G, layout)
# nx.draw_networkx_labels(G, layout)


# pos = graphviz_layout(G)
# twopi, tred, fdp, gvcolor, sfdp, gvpr, unflatten, patchwork, circo, gc, nop, sccmap, ccomps, neato, dot, osage, acyclic
pos = nx.nx_agraph.graphviz_layout(G, prog="twopi")
# nx.draw(G, pos)
# nx.draw(G, pos, with_labels=True)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_labels(G, pos, font_size=10)
for cycle in nx.simple_cycles(G):
    # print(cycle)
    nx.draw_networkx_edges(G, pos, edgelist=[cycle], edge_color="r")
# scale = 1.1
# label_pos = {node: (scale * x, scale * y) for node, (x, y) in pos.items()}
# nx.draw_networkx_labels(G, label_pos, font_size=1)

plt.savefig(f"{curdir}/b.png", bbox_inches="tight", pad_inches=0)
