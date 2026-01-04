import nova
import nova.nn as nn
import nova.nn.eval as eval
import nova.optim as optim
import nova.nn.functional as F
from nova.optim import lr_scheduler
from nova.utils.data import DataLoader
from nova.utils.datasets import load_fashion_mnist_data

(x_train, y_train), (x_test, y_test), (x_val, y_val) = load_fashion_mnist_data(
    tensor4d=True, as_tensor=True, dtype=nova.float32
)

train_loader = DataLoader(x_train, y_train, batch_size=64, shuffle=True)
val_loader = DataLoader(x_val, y_val, batch_size=64, shuffle=True)
test_loader = DataLoader(x_test, y_test, batch_size=64, shuffle=False)

print(f"train -> x: {x_train.size}, y: {y_train.size}")
print(f"train -> x: {x_test.size}, y: {y_test.size}")
print(f"train -> x: {x_val.size}, y: {y_val.size}")


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.14)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.14)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)


class FashionCNN(nn.Module):
    def __init__(self, channel1: int, channel2: int, out_channel: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, channel1, 3, padding=1),
            nn.BatchNorm2d(channel1, momentum=0.12),
            nn.ReLU(),
            ResidualBlock(channel1),
            nn.Conv2d(channel1, channel2, 3, padding=1, stride=2),
            nn.BatchNorm2d(channel2, momentum=0.145),
            nn.ReLU(),
            ResidualBlock(channel2),
            nn.Conv2d(channel2, out_channel, 3, padding=1, stride=2),
            nn.BatchNorm2d(out_channel, momentum=0.18),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channel, 512),
            nn.BatchNorm1d(512, momentum=0.2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.global_avg_pool2d(x)
        x = self.classifier(x)
        return x


model = FashionCNN(channel1=32, channel2=64, out_channel=128)
print(model)

epochs = 10
total_steps = epochs * len(train_loader)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
scheduler = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.005,
    total_steps=total_steps,
    pct_start=0.2,
    div_factor=10,
    final_div_factor=100,
)

print(optimizer)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for input, target in train_loader:
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        model.eval()
        avg_loss = epoch_loss / len(train_loader)
        val_acc = eval.accuracy(model, val_loader)
        print(
            f"Epoch {epoch + 1} - Avg Loss {avg_loss:.4f} - Batch Loss {loss.item():.4f} - Val Acc {val_acc:.4f} - lr {optimizer.param_groups[0]['lr']:.6f}"
        )
