import pytest
import nova
from nova.utils.data import Dataset


class SimpleDataset(Dataset):
    """Simple dataset for testing"""

    def __init__(self, size, input_dim=10, num_classes=2):
        self.data = nova.randn(size, input_dim)
        self.labels = nova.randint(0, num_classes, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class TestDataset:

    def test_dataset_length(self):
        """Test dataset returns correct length"""
        dataset = SimpleDataset(100)
        assert len(dataset) == 100

    def test_single_indexing(self):
        """Test single sample retrieval"""
        dataset = SimpleDataset(50, input_dim=5)
        x, y = dataset[0]
        assert x.shape == (5,)
        assert y.shape == ()

    def test_slice_indexing(self):
        """Test slice retrieval"""
        dataset = SimpleDataset(100, input_dim=5)
        x, y = dataset[0:10]
        assert x.shape == (10, 5)
        assert y.shape == (10,)

    def test_list_indexing(self):
        """Test indexing with list of indices"""
        dataset = SimpleDataset(100, input_dim=5)
        indices = [1, 5, 10, 20]
        x, y = dataset[indices]
        assert x.shape == (4, 5)
        assert y.shape == (4,)

    def test_tensor_indexing(self):
        """Test indexing with tensor"""
        dataset = SimpleDataset(100, input_dim=5)
        indices = nova.tensor([0, 2, 4, 6], dtype=nova.long)
        x, y = dataset[indices]
        assert x.shape == (4, 5)
        assert y.shape == (4,)

    def test_abstract_methods_raise(self):
        """Test that base Dataset raises NotImplementedError"""
        dataset = Dataset()
        with pytest.raises(NotImplementedError):
            len(dataset)
        with pytest.raises(NotImplementedError):
            dataset[0]
