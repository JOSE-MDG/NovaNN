import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load history safely
with open("/home/juancho_col/Documents/NovaNN/training_history.json", "r") as f:
    history = json.load(f)

with open("/home/juancho_col/Documents/NovaNN/pytorch_training_history.json", "r") as f:
    new_history = json.load(f)


# Ensure lists and lengths
acc = history.get("accuracy", [])
loss = history.get("loss", [])
acc2 = new_history.get("accuracy", [])
loss2 = new_history.get("loss", [])
n_epochs = max(len(acc), len(loss), len(acc2), len(loss2))


x = np.arange(1, n_epochs + 1)

# Seaborn theme and palette
sns.set_theme(style="darkgrid", palette="muted")
palette = sns.color_palette("muted")

fig, axes = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)

# Accuracy plot (NovaNN)
axes[0, 0].set_title("NovaNN: Training Accuracy")
sns.lineplot(
    x=x[: len(acc)], y=acc, ax=axes[0, 0], marker="o", color=palette[0], linewidth=2
)
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].set_xlim(1, n_epochs)
axes[0, 0].grid(alpha=0.4)

# annotate final value (project acc)
axes[0, 0].annotate(
    f"{acc[-1]:.4f}",
    xy=(len(acc), acc[-1]),
    xytext=(6, 0),
    textcoords="offset points",
    va="center",
    fontsize=9,
    color=palette[0],
)

# Loss plot (NovaNN)
axes[0, 1].set_title("NovaNN: Training Loss")
sns.lineplot(
    x=x[: len(loss)], y=loss, ax=axes[0, 1], marker="o", color=palette[1], linewidth=2
)
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].set_xlim(1, n_epochs)
axes[0, 1].grid(alpha=0.4)

# annotate final value (project loss)
axes[0, 1].annotate(
    f"{loss[-1]:.4f}",
    xy=(len(loss), loss[-1]),
    xytext=(6, 0),
    textcoords="offset points",
    va="center",
    fontsize=9,
    color=palette[1],
)

# Accuracy plot (PyTorch)
axes[1, 0].set_title("PyTorch: Training Accuracy")
sns.lineplot(
    x=x[: len(acc2)], y=acc2, ax=axes[1, 0], marker="o", color=palette[0], linewidth=2
)
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Accuracy")
axes[1, 0].set_xlim(1, n_epochs)
axes[1, 0].grid(alpha=0.4)

# annotate final value (PyTorch acc)
axes[1, 0].annotate(
    f"{acc2[-1]:.4f}",
    xy=(len(acc2), acc2[-1]),
    xytext=(6, 0),
    textcoords="offset points",
    va="center",
    fontsize=9,
    color=palette[0],
)

# Loss plot (PyTorch)
axes[1, 1].set_title("PyTorch: Training Loss")
sns.lineplot(
    x=x[: len(loss2)], y=loss2, ax=axes[1, 1], marker="o", color=palette[1], linewidth=2
)
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].set_xlim(1, n_epochs)
axes[1, 1].grid(alpha=0.4)

# annotate final value (PyTorch loss)
axes[1, 1].annotate(
    f"{loss2[-1]:.4f}",
    xy=(len(loss2), loss2[-1]),
    xytext=(6, 0),
    textcoords="offset points",
    va="center",
    fontsize=9,
    color=palette[1],
)

sns.despine(trim=True)
plt.show()
