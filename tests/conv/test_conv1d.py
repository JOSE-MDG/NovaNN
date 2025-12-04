# import numpy as np
# import pytest
# from novann.layers import Conv1d
# from novann.utils import numeric_grad_scalar_wrt_x, numeric_grad_wrt_param
# from novann.core import logger


# RNG = np.random.RandomState(0)


# def test_convolutional_forward_shape_and_numeric_backward():
#     # Setup: small convolutional layer with deterministic weight initializer
#     N, in_c, L = 32, 3, 14
#     out_c = 16
#     init_fn = lambda shape: np.ones(shape, dtype=np.float32)
#     conv = Conv1d(in_c, out_c, 3, init=init_fn)

#     # Input: batch of random vectors
#     X = RNG.randn(N, in_c, L)

#     # Forward: shape check and simple numeric expectation (weights are ones)
#     out = conv(X)

#     logger.debug(f"Layer: \n {repr(Conv1d)}")
#     L_out = conv._calc_out_size(L)
#     logger.debug(f"Out length: {L_out}")

#     assert out.shape == (N, out_c, L_out)

#     # Backward: analytic gradient w.r.t. input vs numerical finite differences
#     G = RNG.randn(N, out_c, L_out)

#     logger.debug(f"X shape: \n ({X.shape})")
#     logger.debug(f"Grad shape: \n ({G.shape})")

#     _ = conv(X)
#     dx = conv.backward(G)
#     num_dx = numeric_grad_scalar_wrt_x(lambda z: conv.forward(z), X.copy(), G, eps=1e-6)
#     assert np.allclose(dx, num_dx, atol=1e-4, rtol=1e-3)


# def test_convolutional_param_grads_numeric():
#     # Setup: random (but deterministic) initialization for numeric gradient checks
#     N, in_c, L = 32, 3, 14
#     out_c = 16
#     init_fn = lambda shape: RNG.randn(*shape).astype(np.float32)
#     conv = Conv1d(in_c, out_c, 3, init=init_fn)

#     logger.debug(f"Layer: \n {repr(Conv1d)}")
#     L_out = conv._calc_out_size(L)
#     logger.debug(f"Out length: {L_out}")

#     X = RNG.randn(N, in_c, L).astype(np.float32)
#     G = RNG.randn(N, out_c, L_out).astype(np.float32)

#     logger.debug(f"X shape: \n ({X.shape})")
#     logger.debug(f"Grad shape: \n ({G.shape})")

#     # Ensure forward caches are available and compute analytic gradients
#     _ = conv(X)
#     _ = conv.backward(G)
#     analytic_w_grad = conv.weight.grad.copy()
#     analytic_b_grad = conv.bias.grad.copy()

#     # Numeric gradients by perturbing parameters directly
#     num_w_grad = numeric_grad_wrt_param(conv, "weight", X.copy(), G, eps=1e-6)
#     num_b_grad = numeric_grad_wrt_param(conv, "bias", X.copy(), G, eps=1e-6)

#     # Compare analytic vs numeric parameter gradients
#     assert np.allclose(analytic_w_grad, num_w_grad, atol=1e-6, rtol=1e-5)
#     assert np.allclose(analytic_b_grad, num_b_grad, atol=1e-6, rtol=1e-5)
