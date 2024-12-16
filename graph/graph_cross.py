import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    matrix,
    labels,
    std_matrix=None,
    title="Confusion Matrix",
    cmap="Blues",
    vmin=0,
    vmax=1,
):
    """
    Plot a confusion matrix with color normalization.

    Parameters:
    matrix : numpy.ndarray
        3x3 matrix containing confusion matrix values
    labels : list
        List of string labels for both axes
    title : str
        Title for the plot
    cmap : str
        Colormap for the plot
    vmin : float
        Minimum value for the colormap
    vmax : float
        Maximum value for the colormap
    """

    # Normalize the matrix
    # norm_matrix = matrix.astype("float") / matrix.max()
    norm_matrix = matrix.astype("float")

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create heatmap
    im = ax.imshow(norm_matrix, cmap=cmap, vmin=vmin, vmax=vmax)

    # Show all ticks and label them
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontdict={"fontsize": 18})
    ax.set_yticklabels(labels, fontdict={"fontsize": 18})

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_yticklabels(), rotation=90, ha="center", rotation_mode="anchor")
    ax.tick_params(axis="y", which="both", pad=15)

    # Create text annotations
    threshold = 0.7  # Threshold for switching to white text
    for i in range(len(labels)):
        for j in range(len(labels)):
            # Choose text color based on normalized value
            text_color = "white" if np.abs(norm_matrix[i, j]) > threshold else "black"
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontdict={"fontsize": 22},
            )
            # Shift std text down a bit
            if std_matrix is not None:
                ax.text(
                    j,
                    i + 0.1,
                    f"({std_matrix[i, j]:.3f})",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontdict={"fontsize": 18},
                )

    # Set title and labels
    ax.set_xlabel("Detector Model", fontdict={"fontsize": 22, "weight": "bold"})
    ax.set_ylabel("Generator Model", fontdict={"fontsize": 22, "weight": "bold"})

    # Ensure the plot is tight in the figure
    plt.tight_layout()

    return fig, ax


# Sentence Reordering
labels = ["GPT2-XL", "GPT-J-6B", "GPT-neo-2.7B"]
sentence_confusion_mat_xsum = np.array(
    [
        [0.984, 0.947, 0.955],  # GPT2-XL generator
        [0.961, 0.952, 0.942],  # GPT-J-6B generator
        [0.925, 0.904, 0.934],  # GPT-neo-2.7B generator
    ]
)
baseline_confusion_mat_xsum = np.array(
    [
        [0.987, 0.719, 0.909],  # GPT2-XL generator
        [0.762, 0.958, 0.870],  # GPT-J-6B generator
        [0.814, 0.708, 0.989],  # GPT-neo-2.7B generator
    ]
)


sentence_confusion_mat_squad = np.array(
    [
        [0.935, 0.851, 0.887],  # GPT2-XL generator
        [0.872, 0.872, 0.868],  # GPT-J-6B generator
        [0.836, 0.806, 0.894],  # GPT-neo-2.7B generator
    ]
)
baseline_confusion_mat_squad = np.array(
    [
        [0.987, 0.437, 0.749],  # GPT2-XL generator
        [0.756, 0.873, 0.771],  # GPT-J-6B generator
        [0.773, 0.493, 0.959],  # GPT-neo-2.7B generator
    ]
)


sentence_confusion_mat_writing = np.array(
    [
        [0.803, 0.606, 0.672],  # GPT2-XL generator
        [0.744, 0.771, 0.737],  # GPT-J-6B generator
        [0.711, 0.632, 0.768],  # GPT-neo-2.7B generator
    ]
)
baseline_confusion_mat_writing = np.array(
    [
        [0.994, 0.718, 0.892],  # GPT2-XL generator
        [0.847, 0.951, 0.896],  # GPT-J-6B generator
        [0.893, 0.744, 0.989],  # GPT-neo-2.7B generator
    ]
)


# Get per cell avg and std of all three matrices
sentence_avg_mat = np.mean(
    np.array(
        [
            sentence_confusion_mat_xsum,
            sentence_confusion_mat_squad,
            sentence_confusion_mat_writing,
        ]
    ),
    axis=0,
)
sentence_std_mat = np.std(
    np.array(
        [
            sentence_confusion_mat_xsum,
            sentence_confusion_mat_squad,
            sentence_confusion_mat_writing,
        ]
    ),
    axis=0,
)


# Graph confusion matrix for all datasets
fig, ax = plot_confusion_matrix(
    sentence_avg_mat, labels, std_matrix=sentence_std_mat, title="Sentence Reordering"
)
plt.savefig("cross_sentence_reordering.png")

baseline_avg_mat = np.mean(
    np.array(
        [
            baseline_confusion_mat_xsum,
            baseline_confusion_mat_squad,
            baseline_confusion_mat_writing,
        ]
    ),
    axis=0,
)
baseline_std_mat = np.std(
    np.array(
        [
            baseline_confusion_mat_xsum,
            baseline_confusion_mat_squad,
            baseline_confusion_mat_writing,
        ]
    ),
    axis=0,
)

fig, ax = plot_confusion_matrix(
    baseline_avg_mat,
    labels,
    std_matrix=baseline_std_mat,
    title="Baseline",
    cmap="Blues",
    vmin=0,
    vmax=1,
)
plt.savefig("cross_baseline.png")


# Graph delta for all datasets
mat_xsum_delta = sentence_confusion_mat_xsum - baseline_confusion_mat_xsum
mat_squad_delta = sentence_confusion_mat_squad - baseline_confusion_mat_squad
mat_writing_delta = sentence_confusion_mat_writing - baseline_confusion_mat_writing

delta_avg_mat = np.mean(
    np.array(
        [
            mat_xsum_delta,
            mat_squad_delta,
            mat_writing_delta,
        ]
    ),
    axis=0,
)
delta_std_mat = np.std(
    np.array(
        [
            mat_xsum_delta,
            mat_squad_delta,
            mat_writing_delta,
        ]
    ),
    axis=0,
)

fig, ax = plot_confusion_matrix(
    delta_avg_mat,
    labels,
    std_matrix=delta_std_mat,
    title="Sentence Reordering",
    cmap="RdBu",
    vmin=-0.5,
    vmax=0.5,
)
plt.savefig("cross_sentence_reordering_delta.png")

# Graph delta and confusion matrix without writingprompts
delta_avg_mat = np.mean(
    np.array(
        [
            mat_xsum_delta,
            mat_squad_delta,
        ]
    ),
    axis=0,
)
delta_std_mat = np.std(
    np.array(
        [
            mat_xsum_delta,
            mat_squad_delta,
        ]
    ),
    axis=0,
)

avg_mat = np.mean(
    np.array(
        [
            sentence_confusion_mat_xsum,
            sentence_confusion_mat_squad,
        ]
    ),
    axis=0,
)
std_mat = np.std(
    np.array(
        [
            sentence_confusion_mat_xsum,
            sentence_confusion_mat_squad,
        ]
    ),
    axis=0,
)

fig, ax = plot_confusion_matrix(
    avg_mat,
    labels,
    std_matrix=std_mat,
    title="Sentence Reordering",
    cmap="Blues",
    vmin=0,
    vmax=1,
)
plt.savefig("cross_sentence_reordering_no_writing.png")

fig, ax = plot_confusion_matrix(
    delta_avg_mat,
    labels,
    std_matrix=delta_std_mat,
    title="Sentence Reordering",
    cmap="RdBu",
    vmin=-0.5,
    vmax=0.5,
)
plt.savefig("cross_sentence_reordering_delta_no_writing.png")


# Graph confusion matrix for writingprompts
fig, ax = plot_confusion_matrix(
    sentence_confusion_mat_writing,
    labels,
    title="WritingPrompts",
    cmap="Blues",
    vmin=0,
    vmax=1,
)
plt.savefig("cross_writing.png")

# Graph delta for writingprompts
fig, ax = plot_confusion_matrix(
    mat_writing_delta,
    labels,
    title="WritingPrompts",
    cmap="RdBu",
    vmin=-0.5,
    vmax=0.5,
)
plt.savefig("cross_writing_delta.png")
