import pytest
import io
import pickle
import nova.nn as nn
from pathlib import Path
from nova.serialization import save, load
from nova.exceptions import (
    LoadError,
    UnsafeLoadError,
    FileNotFoundError as NovaFileNotFoundError,
)
from nova.utils import registry_class


class UnregisteredClass:
    """A class that is not registered with NovaNN."""

    def __init__(self):
        self.value = 42


@registry_class
class RegisteredCustomClass:
    """A class registered with NovaNN."""

    def __init__(self, value=10):
        self.value = value


class TestLoadBasic:
    """Test basic load functionality."""

    def test_load_from_file_path(self, tmp_path: Path):
        """Test loading from a file path."""
        model = nn.Linear(10, 5)
        file_path = tmp_path / "model.pth"
        save(model, file_path)

        loaded = load(file_path)

        assert isinstance(loaded, nn.Linear)
        assert loaded.in_features == 10
        assert loaded.out_features == 5

    def test_load_from_buffer(self):
        """Test loading from BytesIO buffer."""
        model = nn.Linear(10, 5)
        buffer = io.BytesIO()
        save(model, buffer)

        buffer.seek(0)
        loaded = load(buffer)

        assert isinstance(loaded, nn.Linear)


class TestLoadSafety:
    """Test safety features of loading."""

    def test_load_unregistered_class_raises_error(self, tmp_path: Path):
        """Test that loading unregistered classes with weights_only=True raises error."""
        obj = UnregisteredClass()
        file_path = tmp_path / "unregistered.pth"

        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

        with pytest.raises(UnsafeLoadError, match="unregistered"):
            load(file_path, weights_only=True)

    def test_load_unregistered_class_unsafe_mode_works(self, tmp_path: Path):
        """Test that loading unregistered classes with weights_only=False works."""
        obj = UnregisteredClass()
        file_path = tmp_path / "unregistered.pth"

        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

        loaded = load(file_path, weights_only=False)

        assert isinstance(loaded, UnregisteredClass)
        assert loaded.value == 42

    def test_load_registered_class_works(self, tmp_path: Path):
        """Test that registered classes load successfully."""
        obj = RegisteredCustomClass(value=100)
        file_path = tmp_path / "registered.pth"
        save(obj, file_path)

        loaded = load(file_path, weights_only=True)

        assert isinstance(loaded, RegisteredCustomClass)
        assert loaded.value


class TestLoadErrors:
    """Test error handling in load."""

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading non-existent file raises FileNotFoundError."""
        with pytest.raises(NovaFileNotFoundError, match="does not exist"):
            load("nonexistent_file.pth")

    def test_load_corrupted_file_raises_error(self, tmp_path: Path):
        """Test that corrupted pickle file raises LoadError."""
        file_path = tmp_path / "corrupted.pth"

        with open(file_path, "wb") as f:
            f.write(b"not a valid pickle")

        with pytest.raises(LoadError):
            load(file_path)

    def test_load_empty_file_raises_error(self, tmp_path: Path):
        """Test that empty file raises LoadError."""
        file_path = tmp_path / "empty.pth"
        file_path.touch()

        with pytest.raises(LoadError):
            load(file_path)


class TestLoadRoundtrip:
    """Test save/load roundtrip."""

    def test_roundtrip_model(self, tmp_path: Path):
        """Test save and load roundtrip for a model."""
        original = nn.Linear(10, 5)
        file_path = tmp_path / "model.pth"

        save(original, file_path)
        loaded = load(file_path)

        assert loaded.in_features == original.in_features
        assert loaded.out_features == original.out_features

    def test_roundtrip_state_dict(self, tmp_path: Path):
        """Test save and load roundtrip for state dict."""
        model = nn.Linear(10, 5)
        state_dict = model.state_dict()
        file_path = tmp_path / "state.pth"

        save(state_dict, file_path)
        loaded = load(file_path)

        assert isinstance(loaded, dict)
        assert "weight" in loaded
        assert "bias" in loaded
