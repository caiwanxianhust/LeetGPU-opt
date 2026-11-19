"""Hand-written specs for kernels the importer could not translate.

tools/import_specs.py refuses to guess: every entry here was rewritten by hand
from the legacy test function named in `test_fn`, keeping the original input
construction, the original torch reference and the original tolerance.  Three
cases needed real work:

* `fused_adamw` -- the legacy test compared `params` with `params` (both
  sides had been rebound to the same clone), so it never checked anything.
  The reference below is the AdamW update the kernel documents, applied to
  copies of the inputs; the mutated tensors are the comparison subjects.
* `moe_topk_gating` -- the legacy test used two different tolerances (1e-3
  for the weights, the dtype-default for the integer indices).  The Spec
  carries one tolerance; the indices are int32 and therefore compared exactly
  regardless of atol/rtol, so 1e-3 loses nothing.
* `count_equal`, `count_equal_2D` -- the legacy assertion was
  `out.item() == torch_out` against a python int.  The references return a
  0-dim int32 tensor instead, which is the same value in the kernel's dtype.
* `multi_agent_sim` -- the legacy reference fills `torch_out` element by
  element inside a loop (`torch_out[i, 2:] = ...`); a subscript store is not a
  binding, so the importer's dependency walk would only see the initial zeros.
  The reference below replays the loop in the original order.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import register
from .spec import Spec


def _shape_of(inputs):
    parts = []
    for arg in inputs[0]:
        if torch.is_tensor(arg):
            parts.append("x".join(str(d) for d in tuple(arg.shape)) or "scalar")
    return " ".join(parts) or "-"


# --- count_equal (from test.py:count_array_elements) -----------------------

def _count_equal_inputs(seed):
    x = torch.randint(-127, 127, (1025,), device="cuda", dtype=torch.int32)
    return (x, 0), {}


def _count_equal_reference(x, K):
    return (x == K).sum().to(torch.int32)


register(Spec(
    name="count_equal",
    test_fn="count_array_elements",
    dtype=torch.int32,
    make_inputs=_count_equal_inputs,
    reference=_count_equal_reference,
    tol=(1e-3, 1e-5),
    shape=_shape_of,
    notes="hand-written: legacy test asserted out.item() == python int",
))


# --- count_equal_2D (from test.py:count_array_elements_2D) ------------------

def _count_equal_2D_inputs(seed):
    x = torch.randint(-127, 127, (1024, 10000), device="cuda", dtype=torch.int32)
    return (x.flatten(), 1024, 10000, 0), {}


def _count_equal_2D_reference(flat, M, N, K):
    return (flat.view(M, N) == K).sum().to(torch.int32)


register(Spec(
    name="count_equal_2D",
    test_fn="count_array_elements_2D",
    dtype=torch.int32,
    make_inputs=_count_equal_2D_inputs,
    reference=_count_equal_2D_reference,
    tol=(1e-3, 1e-5),
    shape=_shape_of,
    notes="hand-written: legacy test asserted out.item() == python int",
))


# --- fused_adamw (from test.py:fused_adamw) --------------------------------

_ADAMW_N, _ADAMW_T = 65536, 100


def _fused_adamw_inputs(seed):
    params = torch.randn(_ADAMW_N, device="cuda", dtype=torch.float32)
    grad = torch.randn(_ADAMW_N, device="cuda", dtype=torch.float32)
    m = torch.zeros(_ADAMW_N, device="cuda", dtype=torch.float32)
    v = torch.zeros(_ADAMW_N, device="cuda", dtype=torch.float32)
    return (params, grad, m, v, 1e-3, 0.9, 0.999, 1e-8, 0.01, _ADAMW_T), {}


def _fused_adamw_reference(params, grad, m, v, lr, beta1, beta2, eps,
                           weight_decay, t):
    # The kernel mutates params/m/v in place, so the reference does too; the
    # harness hands both sides their own copies.
    params.mul_(1.0 - lr * weight_decay)
    m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    bias_c1 = 1.0 - beta1 ** t
    bias_c2 = 1.0 - beta2 ** t
    denom = v.div(bias_c2).sqrt_().add_(eps)
    params.addcdiv_(m, denom, value=-lr / bias_c1)
    return params, m, v


register(Spec(
    name="fused_adamw",
    test_fn="fused_adamw",
    dtype=torch.float32,
    make_inputs=_fused_adamw_inputs,
    reference=_fused_adamw_reference,
    tol=(1e-3, 1e-3),
    shape=_shape_of,
    inplace_indices=(0, 2, 3),
    notes="hand-written: legacy test compared params with params (vacuous)",
))



# --- moe_topk_gating (from test.py:moe_topk_gating) -------------------------

_MOE_M, _MOE_E, _MOE_K = 200, 200, 4


def _moe_inputs(seed):
    logits = torch.rand(_MOE_M, _MOE_E, device="cuda", dtype=torch.float32)
    return (logits, _MOE_K), {}


def _moe_reference(logits, k):
    M = logits.shape[0]
    weights = torch.zeros(M, k, device="cuda", dtype=torch.float32)
    indices = torch.zeros(M, k, device="cuda", dtype=torch.int32)
    for i in range(M):
        vals, idx = torch.topk(logits[i], k)
        indices[i] = idx
        weights[i] = F.softmax(vals, dim=0)
    return weights, indices


register(Spec(
    name="moe_topk_gating",
    test_fn="moe_topk_gating",
    dtype=torch.float32,
    make_inputs=_moe_inputs,
    reference=_moe_reference,
    tol=(1e-3, 1e-3),
    shape=_shape_of,
    notes="hand-written: legacy used per-output tolerances; indices are exact ints",
))


# --- matrix_copy (from test.py:matrix_copy) --------------------------------

_MATRIX_COPY_N = 1024


def _matrix_copy_inputs(seed):
    a = torch.randn(_MATRIX_COPY_N, _MATRIX_COPY_N, device="cuda",
                    dtype=torch.float32)
    return (a.flatten(), _MATRIX_COPY_N), {}


def _matrix_copy_reference(flat, rows):
    return flat.clone()


register(Spec(
    name="matrix_copy",
    test_fn="matrix_copy",
    dtype=torch.float32,
    make_inputs=_matrix_copy_inputs,
    reference=_matrix_copy_reference,
    tol=(1e-5, 1e-5),
    shape=_shape_of,
    notes="hand-written: legacy passed a.flatten() and compared against it",
))


# --- avg_categorical_cross_entropy_loss (from test.py:...) -----------------

_AVG_CE_N, _AVG_CE_C = 9852, 852


def _avg_ce_inputs(seed):
    x = torch.randn(_AVG_CE_N, _AVG_CE_C, device="cuda", dtype=torch.float32)
    y = torch.randint(0, _AVG_CE_C, (_AVG_CE_N,), device="cuda",
                      dtype=torch.int32)
    return (x.view(-1), y, _AVG_CE_N, _AVG_CE_C), {}


def _avg_ce_reference(x_flat, y, N, C):
    x = x_flat.view(N, C)
    log_sum_exp = torch.logsumexp(x, dim=1)
    z_true_class = x.gather(dim=1, index=y.unsqueeze(1).long()).squeeze(1)
    return (log_sum_exp - z_true_class).mean()


register(Spec(
    name="avg_categorical_cross_entropy_loss",
    test_fn="avg_categorical_cross_entropy_loss",
    dtype=torch.float32,
    make_inputs=_avg_ce_inputs,
    reference=_avg_ce_reference,
    tol=(1e-3, 1e-3),
    shape=_shape_of,
    notes="hand-written: kernel takes x.view(-1), reference needs x as (N, C)",
))


# --- color_invert (from test.py:color_inversion) ---------------------------

_COLOR_M, _COLOR_N = 4006, 3997


def _color_invert_inputs(seed):
    x = torch.randint(0, 256, (_COLOR_M, _COLOR_N, 4), device="cuda",
                      dtype=torch.uint8)
    return (x.flatten(), _COLOR_N, _COLOR_M), {}


def _color_invert_reference(flat, width, height):
    pixels = flat.view(height, width, 4)
    out = torch.empty_like(pixels)
    out[:, :, 0] = 255 - pixels[:, :, 0]
    out[:, :, 1] = 255 - pixels[:, :, 1]
    out[:, :, 2] = 255 - pixels[:, :, 2]
    out[:, :, 3] = pixels[:, :, 3]
    return out.flatten()


register(Spec(
    name="color_invert",
    test_fn="color_inversion",
    dtype=torch.uint8,
    make_inputs=_color_invert_inputs,
    reference=_color_invert_reference,
    tol=(1e-3, 1e-3),
    shape=_shape_of,
    inplace_indices=(0,),
    notes="hand-written: kernel inverts the flattened buffer in place",
))


# --- sorting (from test.py:sorting) ---------------------------------------

_SORT_N = 1024 * 128 + 23


def _sorting_inputs(seed):
    x = torch.randn(_SORT_N, device="cuda", dtype=torch.float32)
    return (x.clone(),), {}


def _sorting_reference(values):
    return torch.sort(values)[0]


register(Spec(
    name="sorting",
    test_fn="sorting",
    dtype=torch.float32,
    make_inputs=_sorting_inputs,
    reference=_sorting_reference,
    tol=(1e-3, 1e-3),
    shape=_shape_of,
    inplace_indices=(0,),
    notes="hand-written: the kernel sorts its argument in place",
))


# --- radix_sort (from test.py:radix_sort) ---------------------------------

def _radix_sort_inputs(seed):
    x = torch.randint(0, 2 ** 32, (_SORT_N,), device="cuda", dtype=torch.uint32)
    return (x,), {}


def _radix_sort_reference(x):
    indices = torch.argsort(x.float())
    return torch.gather(x, 0, indices).float()


def _radix_sort_postprocess(value):
    return value.float()


register(Spec(
    name="radix_sort",
    test_fn="radix_sort",
    dtype=torch.uint32,
    make_inputs=_radix_sort_inputs,
    reference=_radix_sort_reference,
    tol=(1e-3, 1e-3),
    shape=_shape_of,
    postprocess=_radix_sort_postprocess,
    notes="hand-written: legacy compared the uint32 result as float32 with "
          "rtol=1e-3, which tolerates order flips between values that collide "
          "once rounded to float32; that leniency is kept, not added",
))


# --- logistic_regression (from test.py:logistic_regression) --------------

_LOGREG_N = 1024 * 128 + 23
_LOGREG_F = 985
_LOGREG_EPOCHS = 50


class _LogReg(torch.nn.Module):
    """Same model the legacy test trained (linear + sigmoid + BCE + SGD)."""

    def __init__(self, n_features, weight_decay=0.01):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_fn = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001,
                                         weight_decay=weight_decay)

    def forward(self, x):
        return self.sigmoid(self.linear(x))


def _logreg_inputs(seed):
    torch.manual_seed(seed)
    x = torch.randn(_LOGREG_N, _LOGREG_F, device="cuda", dtype=torch.float32)
    y = torch.randint(0, 2, (_LOGREG_N,), device="cuda",
                      dtype=torch.int32).float()
    model = _LogReg(_LOGREG_F).to("cuda")
    return (x, y, model.linear.weight.data[0].clone().detach()), {}


def _logreg_reference(x, y, init_beta):
    model = _LogReg(_LOGREG_F).to("cuda")
    with torch.no_grad():
        model.linear.weight[0] = init_beta
    for _ in range(_LOGREG_EPOCHS):
        loss = model.loss_fn(model(x), y.view(-1, 1))
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
    return model.linear.weight.data[0].clone().detach()


register(Spec(
    name="logistic_regression",
    test_fn="logistic_regression",
    dtype=torch.float32,
    make_inputs=_logreg_inputs,
    reference=_logreg_reference,
    tol=(1e-3, 1e-3),
    shape=_shape_of,
    notes="hand-written: reference trains the same 50 SGD/BCE epochs from init_beta",
))


# --- multi_agent_sim (from test.py:multi_agent_simulation) ---------------
#
# The legacy reference fills torch_out element by element inside a loop
# (`torch_out[i, 2:] = ...`), which is a subscript store rather than a
# binding, so the importer's dependency walk cannot see it and would have
# compared the kernel against a zero tensor.  The loop below is the legacy
# loop verbatim, only moved to the CPU: accumulation order stays sequential
# so the float32 rounding matches what the legacy test compared against.

_MAS_N = 500
_MAS_RADIUS2 = 25.0
_MAS_RESPONSE = 0.05


def _multi_agent_inputs(seed):
    return (torch.randn(_MAS_N, 4, device="cuda", dtype=torch.float32),), {}


def _multi_agent_reference(x):
    host = x.detach().to("cpu")
    out = torch.zeros(_MAS_N, 4, dtype=torch.float32)
    for i in range(_MAS_N):
        num_neighbors = 0
        avg_velocity = torch.zeros(2, dtype=torch.float32)
        for j in range(_MAS_N):
            if i == j:
                continue
            distance = host[i, :2] - host[j, :2]
            if float(distance.pow(2).sum()) < _MAS_RADIUS2:
                num_neighbors += 1
                avg_velocity += host[j, 2:]
        if num_neighbors > 0:
            avg_velocity /= num_neighbors
        else:
            avg_velocity = host[i, 2:]
        out[i, 2:] = host[i, 2:] + _MAS_RESPONSE * (avg_velocity - host[i, 2:])
        out[i, :2] = host[i, :2] + out[i, 2:]
    return out.to(x.device)


register(Spec(
    name="multi_agent_sim",
    test_fn="multi_agent_simulation",
    dtype=torch.float32,
    make_inputs=_multi_agent_inputs,
    reference=_multi_agent_reference,
    tol=(1e-4, 1e-4),
    shape=_shape_of,
    notes="hand-written: legacy reference fills the output in a loop of "
          "subscript stores, invisible to the importer's dependency walk",
))


__all__ = [
    "avg_categorical_cross_entropy_loss",
    "multi_agent_sim",
    "color_invert",
    "count_equal",
    "count_equal_2D",
    "fused_adamw",
    "logistic_regression",
    "matrix_copy",
    "moe_topk_gating",
    "radix_sort",
    "sorting",
]
