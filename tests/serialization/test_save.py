import pytest
from pathlib import Path
import io
import pickle
import nova
import nova.nn as nn
from nova import save
from nova.exceptions import SaveError


class TestSaveBasic:
    """Test basic save functionality."""

    def test_save_to_file_path_str(self, tmp_path: Path):
        """Test saving to a string file path."""
        model = nn.Linear(10, 5)
        file_path = tmp_path / "model.pth"

        save(model, str(file_path))

        assert file_path.exists()
        assert file_path.stat().st_size > 0

    def test_save_to_file_path_pathlib(self, tmp_path: Path):
        """Test saving to a Path object."""
        model = nn.Linear(10, 5)
        file_path = tmp_path / "model.pth"

        save(model, file_path)

        assert file_path.exists()
        assert file_path.stat().st_size > 0

    def test_save_to_buffer(self):
        """Test saving to BytesIO buffer."""
        model = nn.Linear(10, 5)
        buffer = io.BytesIO()

        save(model, buffer)

        assert buffer.tell() > 0
        buffer.seek(0)
        loaded = pickle.load(buffer)
        assert isinstance(loaded, nn.Linear)

    def test_save_creates_parent_directories(self, tmp_path: Path):
        """Test that save creates parent directories if they don't exist."""
        model = nn.Linear(10, 5)
        file_path = tmp_path / "subdir" / "nested" / "model.pth"

        save(model, file_path)

        assert file_path.exists()
        assert file_path.parent.exists()


class TestSaveObjects:
    """Test saving different types of objects."""

    def test_save_module(self, tmp_path: Path):
        """Test saving a Module."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        file_path = tmp_path / "model.pth"

        save(model, file_path)
        assert file_path.exists()

    def test_save_tensor(self, tmp_path: Path):
        """Test saving a Tensor."""
        tensor = nova.randn(3, 4)
        file_path = tmp_path / "tensor.pth"

        save(tensor, file_path)
        assert file_path.exists()

    def test_save_state_dict(self, tmp_path: Path):
        """Test saving a state dict."""
        model = nn.Linear(10, 5)
        state_dict = model.state_dict()
        file_path = tmp_path / "state.pth"

        save(state_dict, file_path)
        assert file_path.exists()

    def test_save_dict(self, tmp_path: Path):
        """Test saving a regular dict."""
        data = {"epoch": 10, "loss": 0.5, "weights": nova.randn(5, 5)}
        file_path = tmp_path / "checkpoint.pth"

        save(data, file_path)
        assert file_path.exists()

    def test_save_list_of_tensors(self, tmp_path: Path):
        """Test saving a list of tensors."""
        tensors = [nova.randn(3, 3) for _ in range(5)]
        file_path = tmp_path / "tensors.pth"

        save(tensors, file_path)
        assert file_path.exists()


class TestSaveProtocol:
    """Test different pickle protocols."""

    @pytest.mark.parametrize("protocol", [0, 1, 2, 3, 4, pickle.HIGHEST_PROTOCOL])
    def test_save_with_different_protocols(self, tmp_path: Path, protocol):
        """Test saving with different pickle protocols."""
        model = nn.Linear(10, 5)
        file_path = tmp_path / f"model_protocol_{protocol}.pth"

        save(model, file_path, protocol=protocol)
        assert file_path.exists()


class TestSaveErrors:
    """Test error handling in save."""

    def test_save_none_raises_error(self, tmp_path: Path):
        """Test that saving None raises SaveError."""
        file_path = tmp_path / "none.pth"

        with pytest.raises(SaveError, match="Cannot save None"):
            save(None, file_path)

    def test_save_invalid_file_type_raises_error(self):
        """Test that invalid file argument raises TypeError."""
        model = nn.Linear(10, 5)

        with pytest.raises(SaveError, match="Expected file path"):
            save(model, 123)  # Invalid type

    def test_save_to_readonly_directory_raises_error(self, tmp_path: Path):
        """Test that saving to read-only directory raises PermissionError."""
        import os
        import stat

        model = nn.Linear(10, 5)
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()

        # Make directory read-only
        os.chmod(readonly_dir, stat.S_IRUSR | stat.S_IXUSR)

        file_path = readonly_dir / "model.pth"

        try:
            with pytest.raises(SaveError, match="Permission denied"):
                save(model, file_path)
        finally:
            # Restore permissions for cleanup
            os.chmod(readonly_dir, stat.S_IRWXU)

    def test_save_unpicklable_object_raises_error(self, tmp_path: Path):
        """Test that unpicklable objects raise SaveError."""
        file_path = tmp_path / "lambda.pth"

        # Lambda functions can't be pickled
        unpicklable = lambda x: x + 1

        with pytest.raises(SaveError, match="Failed to pickle"):
            save(unpicklable, file_path)
