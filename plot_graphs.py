import pickle
import matplotlib.pyplot as plt
from pathlib import Path


def plot_graph(path="statistics.pickle", without_validate=False, save=False):
    with open(path, "rb") as f:
        stats = pickle.load(f)

    fig, ax = plt.subplots()
    train_loss = stats['train_loss']
    ax.plot(range(len(train_loss)), train_loss, label=f"train loss. min={min(train_loss):.6f}")
    if not without_validate:
        valid_loss = stats['valid_loss']
        ax.plot(range(len(valid_loss)), valid_loss, label=f"valid loss. min={min(valid_loss):.6f}")

    ax.set(xlabel='epoch', ylabel='loss', title='Mean loss per epoch.')
    ax.legend()
    ax.grid()
    if save:
        fig.savefig("loss.png")
    plt.show()

    fig, ax = plt.subplots()
    train_acc1 = stats['train_acc1']
    ax.plot(range(len(train_acc1)), train_acc1, label=f"train acc1. max={max(train_acc1):.4f}")
    if not without_validate:
        valid_acc1 = stats['valid_acc1']
        ax.plot(range(len(valid_acc1)), valid_acc1, label=f"valid acc1. max={max(valid_acc1):.4f}")

    ax.set(xlabel='epoch', ylabel='accuracy', title='Mean accuracy per epoch.')
    ax.legend()
    ax.grid()
    if save:
        fig.savefig("accuracy.png")
    plt.show()


if __name__ == '__main__':
    path_to_stat = Path("checkpoints/statistics.pickle")
    plot_graph(path=path_to_stat, without_validate=True, save=True)
