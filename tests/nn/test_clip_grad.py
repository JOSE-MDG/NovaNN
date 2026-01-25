import pytest
import nova
import nova.nn as nn
import nova.nn.functional as F
import nova.optim as optim
from nova.nn.utils import clip_grad_value_, clip_grad_norm_

nova.manual_seed(8)


class TestClipping:
    """Class for test the different clipping methods"""

    def test_clip_grad_norm(self):
        """Test clip_grad_norm_() method"""

        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1))

        optimizer = optim.SGD(model.parameters(), lr=0.1)

        for _ in range(20):

            x = nova.randn(32, 10)
            y = nova.randn(32, 1)

            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            for param in model.parameters():
                assert nova.all(param.grad < 1.0)
            optimizer.step()

    def test_clip_grad_value(self):
        """Test clip_grad_value_() method"""
        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2))
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        treshold = 0.5

        for _ in range(20):
            x = nova.randn(32, 10)
            y = nova.randn(32, 2)

            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()

            clip_grad_value_(model.parameters(), clip_value=treshold)

            for param in model.parameters():
                assert nova.all(param.grad <= treshold)
                assert nova.all(param.grad >= -treshold)

            optimizer.step()
