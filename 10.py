import networkx as nx
import matplotlib.pyplot as plt
#  Step 1: Simulate a Scholarly Citation Network
# Nodes represent papers, edges represent citations (Paper A cites Paper B)
citation_edges = [
    ("Paper1", "Paper2"),
    ("Paper1", "Paper3"),
    ("Paper2", "Paper3"),
    ("Paper3", "Paper4"),
    ("Paper4", "Paper5"),
    ("Paper5", "Paper1"), 
    ("Paper6", "Paper2"),
    ("Paper7", "Paper2"),
    ("Paper7", "Paper4"),
    ("Paper8", "Paper4"),
]
#  Step 2: Create a directed graph
G = nx.DiGraph()
G.add_edges_from(citation_edges)
#  Step 3: Visualize the graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, arrows=True, font_size=12)
plt.title(" Scholarly Citation Network")
plt.show()
#  Step 4: Compute PageRank
pagerank_scores = nx.pagerank(G, alpha=0.85)
#  Step 5: Display PageRank scores
print("\n PageRank Scores (Influence of Papers):")
sorted_scores = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
for paper, score in sorted_scores:
    print(f"{paper}: {score:.4f}")
