from utils.globals import plt


def plot_scatter(X, Y, title=None):
    fig, ax = plt.subplots(figsize=(11, 9))

    if title is not None:
        plt.title(title)

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=Y,
        s=50,
    )

    legend = ax.legend(
        *scatter.legend_elements(),
        loc='best',
        title='Classes',
    )
    ax.add_artist(legend)

    plt.show()
