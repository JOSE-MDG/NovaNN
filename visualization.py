import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load history safely

with open("training_history.json", "r") as f:
    history = json.load(f)

# Ensure lists and lengths
acc = history.get("accuracy", [])
loss = history.get("loss", [])
n_epochs = max(len(acc), len(loss))


x = np.arange(1, n_epochs + 1)

# Seaborn theme and palette
sns.set_theme(style="whitegrid", palette="muted")
palette = sns.color_palette("muted")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy plot
axes[0].set_title("Training Accuracy")
sns.lineplot(
    x=x[: len(acc)], y=acc, ax=axes[0], marker="o", color=palette[0], linewidth=2
)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].set_xlim(1, n_epochs)
axes[0].grid(alpha=0.4)

# annotate final value
axes[0].annotate(
    f"{acc[-1]:.4f}",
    xy=(len(acc), acc[-1]),
    xytext=(6, 0),
    textcoords="offset points",
    va="center",
    fontsize=9,
    color=palette[0],
)

# Loss plot
axes[1].set_title("Training Loss")
sns.lineplot(
    x=x[: len(loss)], y=loss, ax=axes[1], marker="o", color=palette[1], linewidth=2
)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_xlim(1, n_epochs)
axes[1].grid(alpha=0.4)

# annotate final value
axes[1].annotate(
    f"{loss[-1]:.4f}",
    xy=(len(loss), loss[-1]),
    xytext=(6, 0),
    textcoords="offset points",
    va="center",
    fontsize=9,
    color=palette[1],
)

sns.despine(trim=True)
plt.tight_layout()
plt.show()
