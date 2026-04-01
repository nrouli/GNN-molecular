"""
Visualize 4 sample molecules from the QM9 dataset.
Each molecule is rendered as a 3D ball-and-stick plot.

Usage:
    python visualize_qm9.py
"""
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from torch_geometric.datasets import QM9
from collections import Counter

# ---------------------------------------------------------------------------
# Atom styling
# ---------------------------------------------------------------------------

ATOM_COLORS = {
    1: "#DDDDDD",   # H  — light gray (visible on white background)
    6: "#333333",   # C  — dark gray
    7: "#3050F8",   # N  — blue
    8: "#FF0D0D",   # O  — red
    9: "#90E050",   # F  — green
}

ATOM_SIZES = {
    1: 120,   # H  — small
    6: 300,   # C  — medium
    7: 280,   # N  — medium
    8: 280,   # O  — medium
    9: 260,   # F  — medium
}

ATOM_NAMES = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
}


# ---------------------------------------------------------------------------
# Plot one molecule
# ---------------------------------------------------------------------------

def plot_molecule(ax, pos, z, edge_index, title=""):
    """
    Draw a 3D ball-and-stick molecule.
    Args:
        ax: matplotlib 3D axis
        pos: (N, 3) atomic positions
        z: (N,) atomic numbers
        edge_index: (2, E) bond connectivity from the molecular graph
        title: subplot title
    """
    pos = pos.numpy()
    z = z.numpy()
    n = len(z)

    # Center the molecule
    pos = pos - pos.mean(axis=0)

    # Draw bonds from the molecular graph
    edge_index_np = edge_index.numpy()
    for k in range(edge_index_np.shape[1]):
        i, j = edge_index_np[0, k], edge_index_np[1, k]
        if i < j:  # avoid drawing each bond twice
            ax.plot(
                [pos[i, 0], pos[j, 0]],
                [pos[i, 1], pos[j, 1]],
                [pos[i, 2], pos[j, 2]],
                color="#888888", linewidth=1.5, alpha=0.7
            )

    # Draw atoms
    for i in range(n):
        atom_z = int(z[i])
        color = ATOM_COLORS.get(atom_z, "#CCCCCC")
        size = ATOM_SIZES.get(atom_z, 200)
        ax.scatter(
            pos[i, 0], pos[i, 1], pos[i, 2],
            c=color, s=size, edgecolors="#444444",
            linewidths=0.5, alpha=0.95, depthshade=True
        )

    # Formatting
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlabel("x (Å)", fontsize=7, labelpad=-10)
    ax.set_ylabel("y (Å)", fontsize=7, labelpad=-10)
    ax.set_zlabel("z (Å)", fontsize=7, labelpad=-10)
    ax.tick_params(axis='both', which='major', labelsize=6)

    max_range = np.abs(pos).max() + 0.5
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(True, alpha=0.2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load dataset WITHOUT any transforms to preserve original bonds
    dataset = QM9(root='data/QM9')

    # Pick 4 molecules of varying size
    torch.manual_seed(42)
    indices = [0, 1000, 5000, 50000]

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle("QM9 Dataset — Sample Molecules", fontsize=16, fontweight='bold', y=0.95)

    mol = dataset[5000]
    print(mol.edge_index)
    print(mol.edge_attr)
    print(mol.pos)
    print(mol.z)

    for i, idx in enumerate(indices):
        mol = dataset[idx]
        pos = mol.pos
        z = mol.z
        n_atoms = len(z)

        # Build formula
        counts = Counter(int(a) for a in z)
        formula = ""
        for atom_z in [6, 7, 8, 9, 1]:  # standard order: C, N, O, F, H
            if atom_z in counts:
                formula += f"{ATOM_NAMES[atom_z]}{counts[atom_z] if counts[atom_z] > 1 else ''}"

        # Target properties
        gap = mol.y[0, 4].item()   # HOMO-LUMO gap
        mu = mol.y[0, 0].item()    # dipole moment

        title = (
            f"Molecule #{idx}\n"
            f"{formula} ({n_atoms} atoms)\n"
            f"Gap: {gap:.3f} eV | \u03bc: {mu:.3f} D"
        )

        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        plot_molecule(ax, pos, z, edge_index=mol.edge_index, title=title)
        ax.view_init(elev=20, azim=45 + i * 30)

    # Legend
    legend_elements = []
    for atom_z, name in ATOM_NAMES.items():
        color = ATOM_COLORS[atom_z]
        legend_elements.append(
            plt.scatter([], [], c=color, s=60, edgecolors="#444444",
                       linewidths=0.5, label=name)
        )

    fig.legend(
        handles=legend_elements, loc='lower center',
        ncol=5, fontsize=10, frameon=True,
        bbox_to_anchor=(0.5, 0.01)
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig("qm9_samples.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.show()
    print("\nSaved to qm9_samples.png")


if __name__ == "__main__":
    main()
