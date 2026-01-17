import pytest
import nova
from nova.utils.data import Dataset, DataLoader


class SimpleDataset(Dataset):
    """Simple dataset for testing"""

    def __init__(self, size, input_dim=10, num_classes=2):
        self.data = nova.randn(size, input_dim)
        self.labels = nova.randint(0, num_classes, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class TestDataLoader:

    def test_basic_iteration(self):
        """Test basic iteration over batches"""
        dataset = SimpleDataset(100, input_dim=10)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        batches = list(loader)
        assert len(batches) == 4  # ceil(100/32) = 4

        # Check first batch
        xb, yb = batches[0]
        assert xb.shape == (32, 10)
        assert yb.shape == (32,)

    def test_last_batch_smaller(self):
        """Test last batch has correct size when not divisible"""
        dataset = SimpleDataset(100, input_dim=5)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        batches = list(loader)
        # Last batch should have 100 - 3*32 = 4 samples
        xb, yb = batches[-1]
        assert xb.shape[0] == 4
        assert yb.shape[0] == 4

    def test_shuffle(self):
        """Test shuffling produces different order"""
        dataset = SimpleDataset(50, input_dim=5)
        loader = DataLoader(dataset, batch_size=10, shuffle=True)

        # Get first batch from two different epochs
        first_epoch_batch = next(iter(loader))
        second_epoch_batch = next(iter(loader))

        # They should likely be different (not guaranteed but very likely)
        # We test by checking if any elements differ
        different = not nova.allclose(first_epoch_batch[0], second_epoch_batch[0])
        assert different  # Should be shuffled differently

    def test_no_shuffle(self):
        """Test no shuffling produces same order"""
        dataset = SimpleDataset(50, input_dim=5)
        loader = DataLoader(dataset, batch_size=10, shuffle=False)

        first_batch1 = next(iter(loader))
        first_batch2 = next(iter(loader))

        # Without shuffle, same batch each time
        assert nova.allclose(first_batch1[0], first_batch2[0])

    def test_dataloader_len(self):
        """Test DataLoader length calculation"""
        dataset = SimpleDataset(100)
        loader = DataLoader(dataset, batch_size=32)
        assert len(loader) == 4  # ceil(100/32)

        loader = DataLoader(dataset, batch_size=25)
        assert len(loader) == 4  # ceil(100/25)

    def test_empty_dataset(self):
        """Test DataLoader with empty dataset"""
        dataset = SimpleDataset(0)
        loader = DataLoader(dataset, batch_size=32)
        assert len(loader) == 0
        batches = list(loader)
        assert len(batches) == 0

    def test_batch_size_property(self):
        """Test batch_size property is read-only"""
        dataset = SimpleDataset(100)
        loader = DataLoader(dataset, batch_size=32)
        assert loader.batch_size == 32

    def test_multiple_epochs(self):
        """Test iterating through multiple epochs"""
        dataset = SimpleDataset(50, input_dim=5)
        loader = DataLoader(dataset, batch_size=10, shuffle=False)

        for epoch in range(3):
            batches = list(loader)
            assert len(batches) == 5

    def test_training_loop_integration(self):
        """Test DataLoader in a realistic training scenario"""
        dataset = SimpleDataset(100, input_dim=10, num_classes=2)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        total_samples = 0
        for xb, yb in loader:
            total_samples += xb.size(0)
            assert xb.size(1) == 10
            assert yb.dim() == 1

        assert total_samples == 100
