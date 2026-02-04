import nova
import numpy as np
import pytest
from nova.utils.decorators import no_inplace_op
from nova.autograd.function import Function


@no_inplace_op
class MockAdd(Function):
    @staticmethod
    def forward(ctx, input, other):
        return input + other

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output, grad_output)


def test_forward_output_type():
    x = nova.tensor([1.0, 2.0], requires_grad=True)
    y = nova.tensor([3.0, 4.0], requires_grad=False)

    out = MockAdd.apply(x, y)

    assert isinstance(out, nova.Tensor)
    assert isinstance(out.data, np.ndarray)
    assert out.data.dtype == nova.float32
    assert np.allclose(out.data, np.array([4.0, 6.0], dtype=nova.float32))


def test_no_grad_required():
    x = nova.tensor([1.0, 2.0], requires_grad=False)
    y = nova.tensor([3.0, 4.0], requires_grad=False)

    out = MockAdd.apply(x, y)

    assert out.grad_fn is None
    assert not hasattr(out, "_ctx") or out._ctx is None


def test_dtype_coercion_and_casting():
    x1 = nova.tensor([1.0, 2.0], dtype=nova.double, requires_grad=True)
    x2 = nova.tensor([3.0, 4.0], dtype=nova.int, requires_grad=True)
    x3 = nova.tensor([5.0, 6.0], dtype=nova.long, requires_grad=True)

    y1 = 3.5  # float
    y2 = 2  # int
    y3 = 7  # int

    out1 = MockAdd.apply(x1, y1)
    out2 = MockAdd.apply(x2, y2)
    out3 = MockAdd.apply(x3, y3)

    outputs = [out1, out2, out3]
    dtypes = [nova.double, nova.int, nova.long]

    for out, dtype in zip(outputs, dtypes):
        assert (
            out.dtype == dtype
        ), f"The dtype of tensor '{out}' is different from {dtype}"


def test_process_containers_and_index_like():
    from nova.autograd.utils import ArgumentProcessor

    x = nova.tensor([1.0], dtype=nova.float32, requires_grad=True)
    args = ([x, 3], {"y": [1, 2, 3]})
    processor = ArgumentProcessor(base_dtype=nova.float32)
    raw_args, raw_kwargs = processor.process_args(args[:1], args[1])

    assert x in processor.get_tracked_tensors()
    assert isinstance(raw_kwargs["y"], list)
    assert isinstance(raw_args[0][0], np.ndarray)
