import pytest
import nova
import numpy as np
from nova.utils.hooks import HooksHandle
from nova.optim import SGD
from nova.nn import Parameter

nova.manual_seed(8)


class TestHooksHandle:

    def test_hook_creation(self):
        """Test HooksHandle creation"""
        hooks_list = []

        def hook_func(x):
            return x * 2

        handle = HooksHandle(hooks_list, hook_func)

        assert handle.hooks_list == hooks_list
        assert handle.hooks_func == hook_func
        assert not handle._removed

    def test_hook_removal(self):
        """Test removing a hook from list"""
        hooks_list = []

        def hook_func(x):
            return x * 2

        hooks_list.append(hook_func)
        handle = HooksHandle(hooks_list, hook_func)

        assert hook_func in hooks_list
        handle.remove()
        assert hook_func not in hooks_list
        assert handle._removed

    def test_multiple_removals(self):
        """Test removing same hook multiple times (should be safe)"""
        hooks_list = []

        def hook_func(x):
            return x * 2

        hooks_list.append(hook_func)

        handle = HooksHandle(hooks_list, hook_func)
        handle.remove()
        handle.remove()

        assert hook_func not in hooks_list


class TestTensorHooks:

    def test_register_backward_hook(self):
        """Test registering backward hook on tensor"""
        x = nova.tensor([1.0, 2.0, 3.0], requires_grad=True)

        hook_called = []

        def my_hook(grad):
            hook_called.append(True)
            return grad * 2

        handle = x.register_hook(my_hook)  # noqa: F841

        # Backward pass
        y = (x**2).sum()
        y.backward()

        assert len(hook_called) == 1
        # Gradient should be doubled: 2*x * 2 = [4, 8, 12]
        assert np.allclose(x.grad, np.array([4.0, 8.0, 12.0]))

    def test_remove_backward_hook(self):
        """Test removing backward hook"""
        x = nova.tensor([1.0, 2.0], requires_grad=True)

        def my_hook(grad):
            return grad * 2

        handle = x.register_hook(my_hook)
        handle.remove()

        y = (x**2).sum()
        y.backward()

        # Hook removed, gradient should be normal
        assert np.allclose(x.grad, np.array([2.0, 4.0]))

    def test_multiple_hooks(self):
        """Test multiple hooks on same tensor"""
        x = nova.tensor([1.0], requires_grad=True)

        calls = []

        def hook1(grad):
            calls.append("hook1")
            return grad

        def hook2(grad):
            calls.append("hook2")
            return grad

        x.register_hook(hook1)
        x.register_hook(hook2)

        y = x**2
        y.backward()

        assert len(calls) == 2
        assert "hook1" in calls
        assert "hook2" in calls


class TestOptimizerHooks:

    def test_register_step_pre_hook(self):
        """Test pre-step hook on optimizer"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1)

        hook_called = []

        def pre_hook(opt):
            hook_called.append("pre")

        handle = optimizer.register_step_prev_hook(pre_hook)  # noqa: F841

        p.grad = np.ones_like(p.data)
        optimizer.step()

        assert len(hook_called) == 1
        assert hook_called[0] == "pre"

    def test_register_step_post_hook(self):
        """Test post-step hook on optimizer"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1)

        hook_called = []

        def post_hook(opt):
            hook_called.append("post")

        handle = optimizer.register_step_post_hook(post_hook)  # noqa: F841

        p.grad = np.ones_like(p.data)
        optimizer.step()

        assert len(hook_called) == 1
        assert hook_called[0] == "post"

    def test_hook_execution_order(self):
        """Test pre-hook executes before post-hook"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1)

        execution_order = []

        def pre_hook(opt):
            execution_order.append("pre")

        def post_hook(opt):
            execution_order.append("post")

        optimizer.register_step_prev_hook(pre_hook)
        optimizer.register_step_post_hook(post_hook)

        p.grad = np.ones_like(p.data)
        optimizer.step()

        assert execution_order == ["pre", "post"]

    def test_remove_optimizer_hook(self):
        """Test removing optimizer hook"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1)

        hook_called = []

        def my_hook(opt):
            hook_called.append(True)

        handle = optimizer.register_step_prev_hook(my_hook)
        handle.remove()

        p.grad = np.ones_like(p.data)
        optimizer.step()

        assert len(hook_called) == 0
