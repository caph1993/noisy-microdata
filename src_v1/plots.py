import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .embedding import MixedColumnGroupEmbedding, CategoricalColumnGroupEmbedding


def plot_changes(
    inpData: np.ndarray,
    sanData: np.ndarray,
    embedding: MixedColumnGroupEmbedding,
    plot_pairs="horizontal",
    show=True,
):
    if plot_pairs == "vertical":
        ncols = min(5, embedding.n_columns)
        nrows = 1 + (embedding.n_columns - 1) // ncols
        fig, axes = plt.subplots(nrows * 2, ncols, figsize=(3.2 * ncols, 5 * nrows))
        axes_up, axes_down = (
            np.ravel(axes).reshape(nrows, 2, ncols).transpose(1, 0, 2).reshape(2, -1)
        )
    else:
        ncols = min(4, embedding.n_columns)
        nrows = 1 + (embedding.n_columns - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols * 2, figsize=(6.4 * ncols, 2.5 * nrows))
        axes_up, axes_down = (
            np.ravel(axes).reshape(nrows, ncols, 2).transpose(2, 0, 1).reshape(2, -1)
        )
    i = 0
    for gr in embedding.groups:
        j = i + gr.n_columns
        while i < j:
            axU = axes_up[i]
            axD = axes_down[i]
            if isinstance(gr, CategoricalColumnGroupEmbedding):
                # Plot a matrix of input output counts for each category
                X_cat = inpData[: len(sanData), i].astype(int)
                Y_cat = sanData[:, i].astype(int)
                counts = np.zeros((gr.n_cat, gr.n_cat))
                for x, y in zip(X_cat, Y_cat):
                    counts[x, y] += 1
                sns.heatmap(counts, ax=axU, annot=False, cbar=True, cmap="Blues")
                X_unique, X_counts = np.unique(X_cat, return_counts=True)
                sns.barplot(
                    x=X_unique,
                    y=X_counts,
                    ax=axD,
                    color="tab:blue",
                    label="inp",
                    alpha=0.5,
                )
                Y_unique, Y_counts = np.unique(Y_cat, return_counts=True)
                sns.barplot(
                    x=Y_unique,
                    y=Y_counts,
                    ax=axD,
                    color="tab:orange",
                    label="san",
                    alpha=0.5,
                )
                axD.legend()
            else:
                diffs = sanData[:, i] - inpData[: len(sanData), i]
                sns.histplot(diffs, ax=axU, kde=False, bins=30, color="tab:blue")
                sns.histplot(
                    inpData[: len(sanData), i],
                    ax=axD,
                    kde=False,
                    color="tab:blue",
                    label="inp",
                    alpha=0.5,
                )
                sns.histplot(
                    sanData[:, i],
                    ax=axD,
                    kde=False,
                    color="tab:orange",
                    label="san",
                    alpha=0.5,
                )
                axD.legend()
            axU.set_title(f"Changes in column {i}")
            i += 1
    while i < len(axes_up):
        axes_up[i].axis("off")
        axes_down[i].axis("off")
        i += 1
    plt.tight_layout()
    # Decorate pairs of axes
    for i in range(embedding.n_columns):
        axU = axes_up[i]
        axD = axes_down[i]
        # x0, *_, x1 = sorted([axU.get_position().x0, axD.get_position().x0, axU.get_position().x1, axD.get_position().x1])
        # y0, *_, y1 = sorted([axU.get_position().y0, axD.get_position().y0, axU.get_position().y1, axD.get_position().y1])
        rend = fig.canvas.get_renderer()  # type:ignore
        x0_, y0_, dx0, dy0 = axU.get_tightbbox(rend).bounds
        x1_, y1_, dx1, dy1 = axD.get_tightbbox(rend).bounds
        x0, x1 = [min(x0_, x1_), max(x0_ + dx0, x1_ + dx1)]
        y0, y1 = [min(y0_, y1_), max(y0_ + dy0, y1_ + dy1)]
        from matplotlib.patches import Rectangle

        rect = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=0,
            facecolor="lightgray",
            alpha=0.3,
            zorder=-1,
        )
        # from matplotlib.colors import LinearSegmentedColormap, to_rgba
        # cmap_color = [(0.5,0.5,0.5,0.5), (1,1,1,0)]
        # cmap = LinearSegmentedColormap.from_list('gradient', cmap_color)
        # rect = Rectangle((x0, y0), x1-x0, y1-y0, linewidth=0, facecolor=cmap(np.linspace(0, 1, 250)), zorder=-1)
        fig.patches.append(rect)
    if show:
        plt.show()
    return
