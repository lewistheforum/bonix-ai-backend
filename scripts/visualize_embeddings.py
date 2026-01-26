#!/usr/bin/env python3
"""
Embeddings Visualization Script

Generates a 2D t-SNE visualization of the knowledge base embeddings
stored in PostgreSQL to help observe data clusters.

Usage:
    python scripts/visualize_embeddings.py

Output:
    scripts/output/embeddings_visualization.png
"""
import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sqlalchemy import create_engine, text
import json

from app.config import settings


def get_sync_database_url() -> str:
    """Get synchronous database URL"""
    return settings.DATABASE_URL_SYNC


def fetch_embeddings_from_db():
    """
    Fetch all embeddings and metadata from the knowledge_base table.
    
    Returns:
        Tuple of (embeddings array, labels list, metadata list)
    """
    print("Connecting to database...")
    engine = create_engine(get_sync_database_url())
    
    query = text("""
        SELECT 
            _id,
            content,
            embedding,
            metadata
        FROM knowledge_base
        WHERE deleted_at IS NULL
            AND embedding IS NOT NULL
        LIMIT 5000
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()
    
    if not rows:
        print("No embeddings found in database!")
        return None, None, None
    
    print(f"Fetched {len(rows)} embeddings from database")
    
    embeddings = []
    labels = []
    metadata_list = []
    
    for row in rows:
        # Parse embedding (stored as vector in PostgreSQL)
        embedding = row.embedding
        if isinstance(embedding, str):
            # If it's a string representation, parse it
            embedding = json.loads(embedding.replace('[', '').replace(']', '').split(','))
        elif hasattr(embedding, '__iter__'):
            embedding = list(embedding)
        
        embeddings.append(embedding)
        
        # Get metadata for labeling
        metadata = row.metadata or {}
        doc_type = metadata.get("type", "unknown")
        labels.append(doc_type)
        
        metadata_list.append({
            "id": str(row._id),
            "content_preview": row.content[:100] if row.content else "",
            "type": doc_type
        })
    
    return np.array(embeddings), labels, metadata_list


def visualize_with_tsne(
    embeddings: np.ndarray,
    labels: list,
    metadata: list,
    output_path: str,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42
):
    """
    Reduce embeddings to 2D using t-SNE and create visualization.
    
    Args:
        embeddings: Array of embedding vectors
        labels: List of labels for coloring
        metadata: List of metadata dicts
        output_path: Path to save the visualization
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations for t-SNE
        random_state: Random seed for reproducibility
    """
    print(f"Running t-SNE on {len(embeddings)} embeddings...")
    print(f"  Perplexity: {perplexity}")
    print(f"  Iterations: {n_iter}")
    
    # Adjust perplexity if needed
    if len(embeddings) < perplexity * 3:
        perplexity = max(5, len(embeddings) // 3)
        print(f"  Adjusted perplexity to {perplexity} due to small dataset")
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init='pca',
        learning_rate='auto'
    )
    
    embeddings_2d = tsne.fit_transform(embeddings)
    print("t-SNE reduction complete!")
    
    # Get unique labels and assign colors
    unique_labels = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot each cluster
    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color_map[label]],
            label=f"{label} ({sum(mask)})",
            alpha=0.7,
            s=50,
            edgecolors='white',
            linewidths=0.5
        )
    
    # Customize the plot
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title(
        "Knowledge Base Embeddings Visualization (t-SNE)\n"
        f"Total: {len(embeddings)} documents | Clusters: {len(unique_labels)}",
        fontsize=14,
        fontweight='bold'
    )
    
    # Add legend
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        title="Document Types",
        title_fontsize=11,
        fontsize=10
    )
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout to accommodate legend
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to: {output_path}")
    
    # Also save as interactive HTML if plotly is available
    try:
        import plotly.express as px
        import pandas as pd
        
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'label': labels,
            'content': [m['content_preview'] for m in metadata]
        })
        
        fig_interactive = px.scatter(
            df, x='x', y='y', color='label',
            hover_data=['content'],
            title='Knowledge Base Embeddings (Interactive)'
        )
        
        html_path = output_path.replace('.png', '.html')
        fig_interactive.write_html(html_path)
        print(f"Interactive visualization saved to: {html_path}")
    except ImportError:
        print("Note: Install plotly for interactive visualization (pip install plotly)")
    
    plt.close()


def analyze_cluster_statistics(embeddings: np.ndarray, labels: list):
    """
    Print statistics about the embedding clusters.
    
    Args:
        embeddings: Array of embedding vectors
        labels: List of labels
    """
    print("\n" + "="*50)
    print("CLUSTER STATISTICS")
    print("="*50)
    
    unique_labels = list(set(labels))
    
    for label in sorted(unique_labels):
        mask = np.array([l == label for l in labels])
        cluster_embeddings = embeddings[mask]
        
        # Calculate centroid
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Calculate variance (spread)
        variance = np.mean(np.var(cluster_embeddings, axis=0))
        
        # Calculate intra-cluster distance
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        avg_distance = np.mean(distances)
        
        print(f"\n{label}:")
        print(f"  Count: {sum(mask)}")
        print(f"  Avg variance: {variance:.4f}")
        print(f"  Avg distance to centroid: {avg_distance:.4f}")
    
    print("\n" + "="*50)


def main():
    """Main function to run the visualization."""
    print("="*60)
    print("KNOWLEDGE BASE EMBEDDINGS VISUALIZATION")
    print("="*60)
    
    # Output path
    output_dir = ROOT_DIR / "scripts" / "output"
    output_path = str(output_dir / "embeddings_visualization.png")
    
    # Fetch embeddings
    embeddings, labels, metadata = fetch_embeddings_from_db()
    
    if embeddings is None or len(embeddings) == 0:
        print("\nNo embeddings to visualize. Please ingest some documents first.")
        print("Use the /api/v1/rag/knowledge-base/sync endpoint to sync data.")
        return
    
    if len(embeddings) < 5:
        print("\nToo few embeddings for meaningful visualization (need at least 5).")
        return
    
    # Analyze clusters
    analyze_cluster_statistics(embeddings, labels)
    
    # Create visualization
    visualize_with_tsne(embeddings, labels, metadata, output_path)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
