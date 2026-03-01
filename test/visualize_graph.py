"""Visualize the LangGraph RAG pipeline as a diagram."""
import logging
import os

from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

from graph.rag_graph import build_graph
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader

# Build the graph
retriever = Retriever()
model_loader = ModelLoader()
graph = build_graph(retriever, model_loader, max_retries=2)

# Get the graph visualization
graph_viz = graph.get_graph()

# Output Mermaid diagram (can paste into https://mermaid.live)
mermaid = graph_viz.draw_mermaid()
print("\n=== Mermaid Diagram ===")
print(mermaid)

# Save Mermaid to file
with open("graph_diagram.mmd", "w") as f:
    f.write(mermaid)
print("\nSaved Mermaid diagram to graph_diagram.mmd")

# Try to save as PNG
try:
    png_bytes = graph_viz.draw_mermaid_png()
    with open("graph_diagram.png", "wb") as f:
        f.write(png_bytes)
    print("Saved PNG to graph_diagram.png")
except Exception as e:
    print(f"\nCould not generate PNG (optional deps missing): {e}")
    print("You can paste the Mermaid code above into https://mermaid.live to get the image")
