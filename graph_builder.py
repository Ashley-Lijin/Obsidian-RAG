import networkx as nx
import matplotlib.pyplot as plt
from valult_reader import notes

def build_graph(notes: dict) -> nx.DiGraph:
    graph = nx.DiGraph()
    for note_name, data in notes.items():
        graph.add_node(note_name)
        for link in data["links"]:
            if link in notes:
                graph.add_edge(note_name, link)
    return graph


def draw_graph(graph: nx.DiGraph):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(graph, k=0.15, iterations=20)
    nx.draw(graph, pos, with_labels=True, node_size=50, node_color="skyblue", font_size=10, font_weight="bold", arrowsize=10)
    plt.title("Obsidian Vault Graph")
    plt.show()



graph = build_graph(notes)
top = sorted(graph.in_degree(), key=lambda x: x[1], reverse=True)[:5]
print("Most linked-to notes:", top)
draw_graph(graph)