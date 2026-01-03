import nova
import pytest
import numpy as np
from nova.utils.log_config import logger


def test_creation_and_dtype():
    t1 = nova.zeros((2, 3))
    t2 = nova.ones((2, 3), dtype=nova.double)
    t3 = nova.full((2, 2), fill_value=5, dtype=nova.int)
    t4 = nova.arange(5)
    t5 = nova.linspace(0, 1, 5)
    t6 = nova.eye(3)
    t7 = nova.one_hot(nova.arange(3), num_classes=5)
    assert t1.dtype == nova.float32
    assert t2.dtype == nova.double
    assert t3.dtype == nova.int
    assert t4.dtype == nova.long
    assert t5.size[0] == 5
    assert t6.size == (3, 3)
    assert t7.size == (3, 5)


def test_basic_math():
    t = nova.arange(5, dtype=nova.float32)
    assert nova.sqrt(t).dtype == t.dtype
    assert t.sum().dtype == t.dtype
    assert t.mean().dtype == t.dtype
    assert t.var().dtype == t.dtype
    assert nova.std(t).dtype == t.dtype
    assert nova.abs(t).dtype == t.dtype
    assert nova.sign(t).dtype == t.dtype


def test_trig_and_exp_log():
    t = nova.arange(1, 5, dtype=nova.float32)
    assert nova.sin(t).dtype == t.dtype
    assert nova.cos(t).dtype == t.dtype
    assert nova.tan(t).dtype == t.dtype
    assert nova.tanh(t).dtype == t.dtype
    assert nova.exp(t).dtype == t.dtype
    assert nova.log(t).dtype == t.dtype


def test_indexing_and_mask():
    t = nova.arange(6)
    mask = t > 2
    t_masked = t[mask]
    idx = nova.tensor([0, 2, 5])
    t_idx = t[idx]
    assert t_masked.dtype == t.dtype
    assert t_idx.dtype == t.dtype


def test_advanced_ops():
    t1 = nova.arange(4, dtype=nova.float32).reshape(2, 2)
    t2 = nova.arange(4, dtype=nova.float32).reshape(2, 2)
    assert nova.dot(t1, t2).dtype == t1.dtype
    assert nova.det(t1).dtype == t1.dtype
    assert nova.trace(t1).dtype == t1.dtype
    assert nova.inv(t1).dtype == t1.dtype
    assert nova.norm(t1).dtype == t1.dtype


def test_boolean_and_where():
    t = nova.arange(5)
    mask = t > 2
    w = nova.where(mask, t, nova.zeros_like(t))
    assert w.dtype == t.dtype
    assert nova.all(w[mask] > 0)


def test_reduction_ops():
    t = nova.arange(6, dtype=nova.float32).reshape(2, 3)
    assert nova.sum(t, dim=0).dtype == t.dtype
    assert nova.min(t, dim=1).dtype == t.dtype
    assert nova.max(t, dim=1).dtype == t.dtype
    assert nova.mean(t, dim=1).dtype == t.dtype


def test_cat_stack():
    t1 = nova.arange(3)
    t2 = nova.arange(3) + 3
    c = nova.cat([t1, t2], dim=0)
    s = nova.stack([t1, t2], dim=0)
    assert c.size[0] == 6
    assert s.size[0] == 2


def test_edge_scalars_and_none():
    t = nova.arange(3)
    t1 = t + 5
    t2 = t * 2.0
    assert t1.dtype == t.dtype
    assert t2.dtype == t.dtype
