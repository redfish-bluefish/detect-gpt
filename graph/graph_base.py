import matplotlib.pyplot as plt
import numpy as np


def create_bar_chart(
    categories,
    values,
    title="Bar Chart",
    xlabel="Categories",
    ylabel="Values",
    group_labels=None,
    color_palette=None,
    legend=True,
):
    """
    Create a bar chart using matplotlib with multiple values.

    Parameters:
    -----------
    categories : list
        List of category names for x-axis
    values : list of lists
        List containing lists of values for each group
    title : str, optional
        Title of the chart
    xlabel : str, optional
        Label for x-axis
    ylabel : str, optional
        Label for y-axis
    color_palette : list, optional
        List of colors for the bars
    """

    # Set the style
    plt.style.use("classic")

    size = (10, 5)
    if legend:
        size = (10, 7)
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=size)
    if legend:
        plt.subplots_adjust(bottom=0.3)

    # Calculate the positions of bars
    num_groups = len(values)
    num_categories = len(categories)
    bar_width = 0.8 / num_groups
    indices = np.arange(num_categories)

    # If no color palette is provided, use default colors
    if color_palette is None:
        color_palette = plt.cm.Set3(np.linspace(0, 1, num_groups))

    if group_labels is None:
        group_labels = [f"Group {i+1}" for i in range(num_groups)]

    # Create bars for each group
    for i in range(num_groups):
        position = indices + (i * bar_width)
        bar = ax.bar(
            position,
            values[i],
            bar_width,
            label=group_labels[i],
            color=color_palette[i],
            alpha=0.8,
        )
        for j, rect in enumerate(bar):
            height = rect.get_height()

            if values[i][j] >= 0.5:
                y_pos = height + 0.01
            else:
                y_pos = height + (0.5 - values[i][j]) + 0.01

            plt.text(
                rect.get_x() + rect.get_width() / 2.0,
                y_pos,
                "{:.2f}".format(height),
                ha="center",
                va="bottom",
                fontsize=12,
            )

    # Customize the plot
    # ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(xlabel, fontsize=16, weight="bold")
    ax.set_ylabel(ylabel, fontsize=16, weight="bold")

    # Set x-axis ticks
    ax.set_xticks(indices + ((num_groups - 1) * bar_width / 2))
    ax.set_xticklabels(categories)
    plt.tick_params(axis="x", which="both", bottom=False, top=False)

    ax.set_xlim(-0.25, len(categories) - 0.1)
    ax.set_ylim(0.5, 1.02)

    # Add legend
    if legend:
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=num_groups / 2)

    # Add grid
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return fig, ax


categories = ["GPT-2XL", "GPT-J-6B", "GPT-neo-2.7B"]
group_labels = [
    "Gap-filling (1 Perturb)",
    "Gap-filling (10 Perturb)",
    "Sentence Reordering",
    "Token Deletion",
    "Model Rewriting",
]


# XSum
gap_filling_1 = [0.809, 0.760, 0.804]
gap_filling_10 = [0.944, 0.885, 0.954]
token_deletion = [0.486, 0.478, 0.502]
model_rewriting = [0.833, 0.751, 0.797]
sentence_reordering = [0.956, 0.906, 0.885]

fig, ax = create_bar_chart(
    categories=categories,
    values=[
        gap_filling_1,
        gap_filling_10,
        sentence_reordering,
        token_deletion,
        model_rewriting,
    ],
    title="Multi-Group Comparison",
    group_labels=group_labels,
    xlabel="Models",
    ylabel="ROC AUC",
    legend=False,
)

plt.savefig("xsum_bar.png")
plt.cla()

# SQuAD
gap_filling_1 = [0.815, 0.708, 0.747]
gap_filling_10 = [0.957, 0.824, 0.905]
token_deletion = [0.623, 0.593, 0.629]
model_rewriting = [0.841, 0.777, 0.806]
sentence_reordering = [0.895, 0.840, 0.837]

fig, ax = create_bar_chart(
    categories=categories,
    values=[
        gap_filling_1,
        gap_filling_10,
        sentence_reordering,
        token_deletion,
        model_rewriting,
    ],
    title="Multi-Group Comparison",
    group_labels=group_labels,
    xlabel="Models",
    ylabel="ROC AUC",
)

plt.savefig("squad_bar.png")
plt.cla()


# WritingPrompts
gap_filling_1 = [0.803, 0.745, 0.810]
gap_filling_10 = [0.970, 0.911, 0.950]
token_deletion = [0.664, 0.595, 0.662]
model_rewriting = [0.848, 0.800, 0.823]
sentence_reordering = [0.781, 0.737, 0.719]

fig, ax = create_bar_chart(
    categories=categories,
    values=[
        gap_filling_1,
        gap_filling_10,
        sentence_reordering,
        token_deletion,
        model_rewriting,
    ],
    title="Multi-Group Comparison",
    group_labels=group_labels,
    xlabel="Models",
    ylabel="ROC AUC",
)

plt.savefig("writing_bar.png")
plt.cla()
