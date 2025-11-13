import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt

# Sample XML Data (you can replace this with actual XML content)
xml_data = '''
<pages>
    <page name="PageA">
        <link>PageB</link>
        <link>PageC</link>
    </page>
    <page name="PageB">
        <link>PageC</link>
    </page>
    <page name="PageC">
        <link>PageA</link>
    </page>
    <page name="PageD">
        <link>PageC</link>
    </page>
</pages>
'''

# Parse XML
root = ET.fromstring(xml_data)

# Create directed graph
web_graph = nx.DiGraph()

# Add nodes and edges from XML data
for page in root.findall('page'):
    src = page.get('name')
    web_graph.add_node(src)
    for link in page.findall('link'):
        dest = link.text
        web_graph.add_edge(src, dest)

# Display the graph
plt.figure(figsize=(6, 5))
nx.draw(web_graph, with_labels=True, node_color='skyblue', edge_color='gray',
        node_size=2000, font_size=14)
plt.title("Web Graph", fontsize=16)
plt.show()

# Compute PageRank
pagerank_scores = nx.pagerank(web_graph, alpha=0.85)

# Display results
print("PageRank Scores:")
for page, score in pagerank_scores.items():
    print(f"{page}: {score:.4f}")
