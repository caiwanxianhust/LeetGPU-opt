"""Generate harness/specs_legacy.py from the retired test.py.

Every test in test.py has the same shape:

    inputs ...        (torch.randn / randint / zeros on cuda)
    [torch reference computation, possibly before or after the call]
    out = leetgpu.<kernel>(args)        # or a @timeit wrapper around it
    [post-processing of out: out = out.view(N, N)]
    compare_tensors(out, torch_out, atol=..., rtol=...)   # or assert allclose

This tool rewrites one such function into a Spec:

    _make_<k>(seed)   -> (args, kwargs)            # only the statements the
                                                   # call's arguments depend on
    _ref_<k>(inputs)  -> expected                  # only the statements the
                                                   # compared reference values
                                                   # depend on, from rebound args
    _shape_<k>(inputs)

Nothing is executed: the translation is pure AST rewriting, so it is
deterministic and the generated file is reproducible.  A function the rewriter
does not fully understand is *not* guessed at: it is reported as MANUAL and
left for harness/specs_manual.py, which keeps the "no invented references"
rule.

Usage:
    python3 tools/import_specs.py [--test test.py] [--out harness/specs_legacy.py]
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

EXCLUDE = {
    # already hand-written in harness/kernels.py (with a cost model)
    "fused_layernorm_linear_gelu",
    "fused_ln_linear_gelu",
    # legacy test compares params with params: it never checked anything
    "fused_adamw",
}

BUILTINS = {
    "int", "float", "bool", "range", "len", "min", "max", "abs", "sum",
    "tuple", "list", "enumerate", "zip", "print", "round", "divmod",
}

# calls that only exist to time or report, never to compute expected values
NOISE_CALLS = {
    "print", "synchronize", "time", "item", "cpu", "contiguous",
}


@dataclass
class Unit:
    """One legacy test function being translated."""

    test_fn: str
    kernel: str
    body: list[ast.stmt]
    call_idx: int
    args: list[ast.expr]
    kwargs: dict[str, ast.expr]
    targets: list[str] | None      # names the call result is unpacked into
    inplace_arg: int | None        # index into args for bare-call kernels
    compares: list[tuple[ast.expr, ast.expr, dict]] = field(default_factory=list)
    imports: set[str] = field(default_factory=set)
    helpers: dict[str, ast.FunctionDef] = field(default_factory=dict)


class Manual(Exception):
    pass


def _is_leetgpu_call(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "leetgpu")


def _root_name(expr: ast.AST) -> str | None:
    while isinstance(expr, (ast.Attribute, ast.Subscript, ast.Call, ast.Starred)):
        if isinstance(expr, ast.Call):
            expr = expr.func
        elif isinstance(expr, ast.Attribute):
            expr = expr.value
        elif isinstance(expr, ast.Subscript):
            expr = expr.value
        else:
            expr = expr.value
    return expr.id if isinstance(expr, ast.Name) else None


def _loads(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Load)}


def _targets(stmt: ast.stmt) -> list[str]:
    if isinstance(stmt, ast.Assign):
        names: list[str] = []
        for t in stmt.targets:
            names += _assign_names(t)
        return names
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return [stmt.target.id]
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return _assign_names(stmt.target)
    if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
        return [stmt.target.id]
    return []


def _assign_names(t: ast.AST) -> list[str]:
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, (ast.Tuple, ast.List)):
        out: list[str] = []
        for e in t.elts:
            out += _assign_names(e)
        return out
    return []


def _literal_names(stmt: ast.stmt) -> bool:
    """True when re-executing the statement always yields the same value."""
    if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        return False
    value = stmt.value
    for node in ast.walk(value):
        if isinstance(node, ast.Attribute):
            return False
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ("int", "float", "bool")):
                return False
        if isinstance(node, ast.Subscript):
            return False
    return True


def _module_imports(tree: ast.Module) -> dict[str, str]:
    """module-level name -> unparsed import statement that binds it."""
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            src = ast.unparse(node)
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                out[name] = src
    return out


def _walk_shallow(node):
    # walk without descending into nested function bodies
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                        ast.Lambda, ast.ClassDef)):
        return
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _walk_shallow(child)


def _find_call_in(nodes):
    # (call, parent stmt, enclosing def); nested defs are visited on their own
    found = []
    def visit(stmts, inside):
        for stmt in stmts:
            for node in _walk_shallow(stmt):
                if _is_leetgpu_call(node):
                    found.append((node, stmt, inside))
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(stmt.body, stmt)
    visit(nodes, None)
    return found


def _find_compare(stmt: ast.stmt) -> tuple[ast.expr, ast.expr, dict] | None:
    call = None
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
    elif isinstance(stmt, ast.Assert) and isinstance(stmt.test, ast.Call):
        call = stmt.test
    if call is None:
        return None
    fn = call.func
    name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
    if name not in ("compare_tensors", "allclose"):
        return None
    if len(call.args) < 2:
        return None
    kwargs = {k.arg: k.value for k in call.keywords if k.arg}
    for stmt2 in ast.walk(call):
        pass
    return call.args[0], call.args[1], {"fn": name, "call": call, "kwargs": kwargs}


def _tol_from(compare: dict) -> tuple[float, float]:
    """Tolerance, honouring the two incompatible legacy signatures.

    compare_tensors(a, b, atol, rtol): position 3 = atol, 4 = rtol
    torch.allclose(a, b, rtol, atol):  position 3 = rtol, 4 = atol
    """
    call: ast.Call = compare["call"]
    fn = compare["fn"]
    if fn == "allclose":
        kw_default = {"rtol": 1e-5, "atol": 1e-8}
        pos = ["rtol", "atol"]
    else:
        kw_default = {"atol": 1e-3, "rtol": 1e-5}
        pos = ["atol", "rtol"]
    vals = dict(kw_default)
    for i, name in enumerate(pos):
        idx = 2 + i
        if idx < len(call.args):
            vals[name] = _const(call.args[idx])
    for k in call.keywords:
        if k.arg in vals:
            vals[k.arg] = _const(k.value)
    return float(vals["atol"]), float(vals["rtol"])


def _const(expr: ast.AST) -> float:
    try:
        return float(ast.literal_eval(expr))
    except (ValueError, SyntaxError):
        pass
    # allow 1e-3 * 10 style constant folding
    if isinstance(expr, ast.BinOp):
        left, right = _const(expr.left), _const(expr.right)
        if isinstance(expr.op, ast.Mult):
            return left * right
        if isinstance(expr.op, ast.Div):
            return left / right
        if isinstance(expr.op, ast.Add):
            return left + right
        if isinstance(expr.op, ast.Sub):
            return left - right
    raise Manual(f"non-literal tolerance: {ast.unparse(expr)}")


def analyse(fn: ast.FunctionDef, module_helpers=None) -> Unit:
    fn = _inline_helpers(fn, module_helpers or {})
    helpers = dict(module_helpers or {})
    for node in ast.walk(fn):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn \
                and node.name not in helpers:
            helpers[node.name] = node
    body = list(fn.body)
    # hoist docstring away
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]

    sites = _find_call_in(body)
    if not sites:
        raise Manual("no leetgpu call")
    kernels = {c.func.attr for c, _, _ in sites}
    if len(kernels) > 1:
        raise Manual(f"calls several kernels: {sorted(kernels)}")
    kernel = kernels.pop()

    call, stmt, wrapper = sites[0]
    # the call may sit inside a @timeit wrapper: map its parameters to the real
    # arguments at the call site of that wrapper.
    if wrapper is not None:
        uses = [i for i, s in enumerate(body)
                if any(_is_name_call(n, wrapper.name) for n in ast.walk(s))]
        if len(uses) != 1:
            raise Manual("wrapper is not called exactly once")
        call_idx = uses[0]
        outer = _first_call_named(body[call_idx], wrapper.name)
        params = [a.arg for a in wrapper.args.args]
        if wrapper.args.defaults or wrapper.args.kwarg or wrapper.args.vararg:
            raise Manual("wrapper has defaults/*args")
        if len(outer.args) != len(params):
            raise Manual("wrapper arity mismatch")
        args = list(outer.args)
        kwargs = {k.arg: k.value for k in outer.keywords if k.arg}
    else:
        call_idx = body.index(stmt)
        args = list(call.args)
        kwargs = {k.arg: k.value for k in call.keywords if k.arg}

    owner = body[call_idx] if wrapper is not None else stmt
    if isinstance(owner, ast.Expr):
        targets = None
    elif isinstance(owner, ast.Assign):
        targets = []
        for t in owner.targets:
            targets += _assign_names(t)
    elif isinstance(owner, ast.Return):
        targets = None
    else:
        raise Manual(f"call result used inside {type(owner).__name__}")

    inplace = None
    if targets is None:
        # bare call: kernel mutates an argument
        names = [a.id for a in args if isinstance(a, ast.Name)]
        compared = _compare_candidates(body)
        picked = [n for n in names if n in compared]
        if len(picked) != 1:
            raise Manual("bare call but cannot tell which argument is mutated")
        inplace = names.index(picked[0])
        targets = [picked[0]]

    unit = Unit(test_fn=fn.name, kernel=kernel, body=body, call_idx=call_idx,
                args=args, kwargs=kwargs, targets=targets, inplace_arg=inplace,
                helpers=helpers)

    # comparisons, in body order
    for s in body[call_idx + 1:]:
        cmp = _find_compare(s)
        if cmp is not None:
            unit.compares.append(cmp)
    if not unit.compares:
        raise Manual("no compare_tensors/allclose call")
    for cand, ref, meta in unit.compares:
        if ast.dump(cand) == ast.dump(ref):
            raise Manual("vacuous comparison (compares a value with itself)")
    if unit.targets is not None and len(unit.compares) != len(unit.targets):
        raise Manual(f"{len(unit.compares)} comparisons for {len(unit.targets)} outputs")
    if unit.kwargs:
        raise Manual("kernel called with keyword arguments")
    return unit


def _is_name_call(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def _first_call_named(stmt: ast.stmt, name: str) -> ast.Call:
    for node in ast.walk(stmt):
        if _is_name_call(node, name):
            return node
    raise Manual("wrapper call not found")


def _compare_candidates(body: list[ast.stmt]) -> set[str]:
    out: set[str] = set()
    for s in body:
        cmp = _find_compare(s)
        if cmp is not None and isinstance(cmp[0], ast.Name):
            out.add(cmp[0].id)
    return out


def _deep_targets(stmt):
    # names bound anywhere inside stmt, including nested control flow
    names = []
    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.Lambda, ast.ClassDef)):
            return
        if isinstance(node, ast.Assign):
            for t in node.targets:
                names.extend(_assign_names(t))
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            names.extend(_assign_names(node.target))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            names.extend(_assign_names(node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    names.extend(_assign_names(item.optional_vars))
        for child in ast.iter_child_nodes(node):
            visit(child)
    visit(stmt)
    return names


def _inlineable(node):
    # a nested helper we can beta-reduce: pure single-return, fixed arity
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    if node.args.defaults or node.args.kw_defaults or node.args.vararg \
            or node.args.kwarg or node.args.kwonlyargs:
        return False
    if any(_is_leetgpu_call(n) for n in ast.walk(node)):
        return False
    b = [s for s in node.body
         if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    return len(b) == 1 and isinstance(b[0], ast.Return) and b[0].value is not None


def _subst(expr, mapping):
    class S(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in mapping:
                return ast.copy_location(ast.parse(ast.unparse(mapping[node.id]), mode="eval").body, node)
            return node
    return S().visit(expr)


def _inline_pass(fn, helpers):
    class T(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            h = helpers.get(node.func.id) if isinstance(node.func, ast.Name) else None
            if h is None or node.keywords:
                return node
            params = [a.arg for a in h.args.args]
            if len(node.args) != len(params):
                return node
            # Deep-copy the helper body first: _subst rewrites Name nodes in
            # place, and the helper's body node is shared by every call site --
            # substituting once would leak that call's arguments into the rest
            # (elu(Q) silently became elu(K.transpose(0, 1))).
            body = ast.parse(ast.unparse(h.body[-1].value), mode="eval").body
            return _subst(body, dict(zip(params, node.args)))
    out = T().visit(fn)
    return ast.fix_missing_locations(out)


def _inline_helpers(fn, module_helpers=None):
    # beta-reduce nested helper calls so the reference reads as one block
    for _ in range(4):
        helpers = {k: v for k, v in (module_helpers or {}).items() if _inlineable(v)}
        for node in ast.walk(fn):
            if node is not fn and _inlineable(node):
                helpers[node.name] = node
        if not helpers:
            return fn
        before = ast.dump(fn)
        fn = _inline_pass(fn, helpers)
        # drop definitions that are now unused (only if used as a call)
        kept = []
        for s in fn.body:
            if isinstance(s, ast.FunctionDef) and s.name in helpers \
                    and not any(isinstance(n, ast.Name) and n.id == s.name
                                and isinstance(n.ctx, ast.Load)
                                for stmt in fn.body for n in ast.walk(stmt)):
                continue
            kept.append(s)
        fn.body = kept
        if ast.dump(fn) == before:
            return fn
    return fn

def _defs(body: list[ast.stmt]) -> dict[str, list[int]]:
    """name -> every statement index that binds it, in program order.

    A name may be rebound (\`torch_out = softmax(...)\` then
    \`torch_out = matmul(torch_out, v)\`), so resolution has to pick the
    binding that is live *at the point of use*, not the last one in the file.
    """
    index: dict[str, list[int]] = {}
    for i, s in enumerate(body):
        for name in _deep_targets(s):
            index.setdefault(name, []).append(i)
    return index


def _subscript_roots(stmt: ast.stmt) -> set[str]:
    """Names a loop body writes through a subscript (``a[i] = v``).

    Such a store binds nothing, so it is invisible to _deep_targets; the caller
    uses this to notice that the reference's tensor is filled by the loop.
    """
    roots: set[str] = set()

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.Lambda, ast.ClassDef)):
            return
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, (ast.Subscript, ast.Attribute)):
                    root = _root_name(tgt)
                    if root:
                        roots.add(root)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            if isinstance(node.target, (ast.Subscript, ast.Attribute)):
                root = _root_name(node.target)
                if root:
                    roots.add(root)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(stmt)
    return roots


def _closure(names: list[str], body: list[ast.stmt],
             index: dict[str, list[int]], skip: set[int],
             before: int | None = None) -> set[int]:
    """Statement indices the given names depend on (transitively, in order)."""
    seen: set[int] = set()
    limit = len(body) if before is None else before

    def resolve(name: str, at: int) -> None:
        for idx in reversed(index.get(name, ())):
            if idx >= at:
                continue
            if idx in seen or idx in skip:
                return
            seen.add(idx)
            deps = _loads(body[idx])
            stmt = body[idx]
            # "x += y" reads x as well as writing it, but the target is a
            # Store node and therefore invisible to _loads(); pull in the
            # binding that produced the value being augmented.
            if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                deps.add(stmt.target.id)
            for dep in deps:
                resolve(dep, idx)
            return

    for name in names:
        resolve(name, limit)

    # A loop that fills a tensor through subscript stores (``torch_out[i, j] = ...``)
    # binds no new name, so _defs/_loads cannot see that the reference depends on
    # it -- the translated reference would keep the initial zeros.  Pull in such
    # loops only while they write to a name the reference already uses.
    while True:
        touched = set(names)
        for idx in seen:
            touched |= _loads(body[idx]) | set(_deep_targets(body[idx]))
        added = False
        for idx in range(limit):
            if idx in seen or idx in skip:
                continue
            stmt = body[idx]
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            mutated = _subscript_roots(stmt)
            if not mutated & touched:
                continue
            seen.add(idx)
            added = True
            for dep in _loads(stmt) | mutated:
                resolve(dep, idx)
        if not added:
            return seen


def _src(stmt: ast.stmt) -> str:
    return ast.unparse(stmt)


def _indent(text: str, pad: str = "    ") -> str:
    return "\n".join(pad + line if line.strip() else "" for line in text.splitlines())


def _bound_value(name, stmt):
    """The value expression that binds ``name`` in ``stmt```.

    Handles both ``x = 1`` and tuple unpacking ``a, b = 1, 2`` (the target then
    is an ast.Tuple, so the plain ``stmt.value`` is the whole tuple and must
    not be used for one of its elements).
    """
    if not isinstance(stmt, ast.Assign):
        return None
    for target in stmt.targets:
        if isinstance(target, ast.Name) and target.id == name:
            return stmt.value
        if (isinstance(target, (ast.Tuple, ast.List))
                and isinstance(stmt.value, (ast.Tuple, ast.List))
                and len(target.elts) == len(stmt.value.elts)):
            for tgt, val in zip(target.elts, stmt.value.elts):
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return val
    return None


def _eval_const(name, defs, body, stack):
    # resolve a local constant to its value (ints/floats only, no tensors)
    if name in stack:
        raise Manual(f"cyclic constant {name}")
    stmt = body[list(defs[name])[-1]]
    value = _bound_value(name, stmt)
    if value is None:
        raise Manual(f"postprocess needs non-literal {name}")
    for node in ast.walk(value):
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            raise Manual(f"postprocess needs non-literal {name}")
        if isinstance(node, ast.Call) and not (
                isinstance(node.func, ast.Name) and node.func.id in ("int", "float", "bool")):
            raise Manual(f"postprocess needs non-literal {name}")
    env = {}
    for dep in _loads(value):
        if dep in defs:
            env[dep] = _eval_const(dep, defs, body, stack | {name})
    try:
        return eval(ast.unparse(value), {"__builtins__": {}}, env)
    except Exception as exc:
        raise Manual(f"postprocess constant {name}: {exc}")

# Kernels the translator deliberately leaves alone.  Each entry is a hand-written
# Spec in harness/specs_manual.py; the reason is printed by the report.
FORCE_MANUAL = {
    "multi_agent_sim": (
        "the reference builds torch_out element by element inside a loop "
        "(`torch_out[i, 2:] = ...`); a subscript store is not a binding, so the "
        "dependency walk cannot see it and would compare against zeros"
    ),
}


def translate(unit: Unit, module_names: set[str]) -> dict:
    if unit.kernel in FORCE_MANUAL:
        raise Manual(FORCE_MANUAL[unit.kernel])
    body = unit.body
    skip_stmts = {unit.call_idx}
    # comparisons and asserts are not part of the reference
    for i, s in enumerate(body):
        if _find_compare(s) is not None or isinstance(s, ast.Assert):
            skip_stmts.add(i)

    arg_names = [n for n in (_root_name(a) for a in unit.args) if n]
    defs = _defs(body)
    input_idx = _closure(arg_names, body, defs, skip_stmts, before=unit.call_idx)

    ref_exprs = [c[1] for c in unit.compares]
    ref_names = [n for n in _loads(ast.Tuple(elts=ref_exprs, ctx=ast.Load()))
                 if n in defs]
    # A helper that gets hoisted into the reference keeps its own name lookup:
    # whatever it reads from the enclosing test function (``num_q_heads``,
    # ``head_dim``, ...) is a dependency of the reference too, even though the
    # call site itself only mentions the helper's name.
    seen_helpers: set[str] = set()
    ref_idx = _closure(ref_names, body, defs, skip_stmts)
    for _ in range(10):
        used_now = set(ref_names)
        for i in ref_idx:
            used_now |= _loads(body[i])
        pending = [n for n in sorted(used_now)
                   if n in unit.helpers and n not in seen_helpers]
        if not pending:
            break
        added = False
        for name in pending:
            helper = unit.helpers[name]
            seen_helpers.add(name)
            local = {a.arg for a in helper.args.args}
            for hstmt in helper.body:
                local |= set(_deep_targets(hstmt))
            for node in ast.walk(helper):
                if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                        and node.id in defs and node.id not in local
                        and node.id not in ref_names):
                    ref_names.append(node.id)
                    added = True
        if not added:
            break
        ref_idx = _closure(ref_names, body, defs, skip_stmts)
    ref_used = set(ref_names)
    for i in ref_idx:
        ref_used |= _loads(body[i])

    # a name the reference needs but the kernel never received cannot be
    # rebuilt: re-running its constructor would produce a different tensor.
    for name in sorted(ref_used):
        if name in arg_names or name not in defs:
            continue
        stmt_idx = defs[name][-1]
        if stmt_idx in input_idx and not _literal_names(body[stmt_idx]):
            raise Manual(f"reference needs {name!r}, which is not passed to the kernel")

    # the reference rebinds argument names from the call arguments, so an
    # argument that is not a bare local (\`x.view(-1)\`) would make it see the
    # reshaped tensor where the test saw the original one.
    for arg in unit.args:
        root = _root_name(arg)
        if root and not isinstance(arg, ast.Name) and root in ref_used:
            raise Manual(f"reference needs the pre-call value of {root!r}, "
                         f"but the kernel received {ast.unparse(arg)!r}")
    ref_ids = [c[1].id for c in unit.compares if isinstance(c[1], ast.Name)]

    # post-processing: <cand> = <cand>.<op>(...)
    postprocess = {}
    for i, s in enumerate(body):
        if i in skip_stmts or not isinstance(s, ast.Assign):
            continue
        if len(s.targets) != 1 or not isinstance(s.targets[0], ast.Name):
            continue
        name = s.targets[0].id
        if name in ref_ids or name not in _loads(s.value):
            continue
        if _root_name(s.value) != name:
            continue
        postprocess[name] = (i, s.value)

    if postprocess and any(n not in postprocess for n in ref_names):
        pass  # candidate side only; fine

    # ---- build make_inputs body -------------------------------------------
    make_lines: list[str] = []
    for i in sorted(input_idx):
        make_lines.append(_src(body[i]))
    # torch.manual_seed / cuda sync keep determinism
    for i, s in enumerate(body[:unit.call_idx]):
        if i in skip_stmts:
            continue
        txt = _src(s)
        if "manual_seed" in txt and i not in input_idx:
            make_lines.append(txt)
    arg_src = ", ".join(ast.unparse(a) for a in unit.args)
    comma = "," if len(unit.args) == 1 else ""
    make_lines.append(f"return ({arg_src}{comma}), {{}}")

    # ---- build reference body ---------------------------------------------
    rebind = []
    for i, a in enumerate(unit.args):
        rebind.append(_root_name(a) or f"_arg{i}")
    if len(rebind) == 1:
        ref_lines = [f"{rebind[0]}, = args"]
    elif rebind:
        ref_lines = [", ".join(rebind) + " = args"]
    else:
        ref_lines = []
    # An argument whose last binding before the call is a non-literal input
    # statement comes straight from the caller; replaying an earlier literal
    # binding of the same name would shadow the rebinding
    # (``x = [floats]`` before ``x = torch.tensor(x)`` in rms_norm_1D).
    rebound = set()
    for root in arg_names:
        before_call = [idx for idx in defs.get(root, ()) if idx < unit.call_idx]
        if before_call and before_call[-1] in input_idx \
                and not _literal_names(body[before_call[-1]]):
            rebound.add(root)
    for i in sorted(ref_idx):
        if i in input_idx and not _literal_names(body[i]):
            continue  # rebuilt by the rebinding above
        if i in input_idx and set(_deep_targets(body[i])) & rebound:
            continue  # shadowed by the rebinding above
        if i in input_idx and _literal_names(body[i]):
            ref_lines.append(_src(body[i]))
        elif i not in input_idx:
            ref_lines.append(_src(body[i]))

    expected = [ast.unparse(e) for e in ref_exprs]
    ref_lines.append("return " + (", ".join(expected) if len(expected) > 1 else expected[0]))

    # undefined-name guard: hoist legacy helper functions when needed
    defined = set(arg_names) | module_names | BUILTINS | {"inputs", "args"}
    defined |= {f"_arg{i}" for i in range(len(unit.args))}
    for i in sorted(ref_idx):
        defined |= set(_deep_targets(body[i]))
    hoisted: list[str] = []
    for _ in range(50):
        used = set()
        for line in hoisted + ref_lines:
            used |= {n.id for n in ast.walk(ast.parse(line)) if isinstance(n, ast.Name)}
        missing = used - defined
        if not missing:
            break
        fresh = [n for n in sorted(missing) if n in unit.helpers]
        if not fresh:
            raise Manual(f"reference needs names it cannot rebuild: {sorted(missing)}")
        for name in fresh:
            node = unit.helpers[name]
            node = ast.parse(ast.unparse(node)).body[0]  # deep copy
            node.decorator_list = []  # @timeit only adds noise
            hoisted.append(ast.unparse(node))
            defined.add(name)
            defined |= {a.arg for a in node.args.args}
            for sub in node.body:
                defined |= set(_deep_targets(sub))
    else:
        raise Manual("helper resolution did not settle")
    ref_lines = hoisted + ref_lines

    # ---- tolerance ---------------------------------------------------------
    tols = []
    for cand, ref, meta in unit.compares:
        try:
            tols.append(_tol_from(meta))
        except Manual:
            raise Manual("non-literal tolerance")
    if len({t for t in tols}) != 1:
        raise Manual(f"per-output tolerances differ: {tols}")
    tol = tols[0]

    # ---- dtype -------------------------------------------------------------
    dtype = None
    for i in sorted(input_idx):
        node = body[i]
        if not isinstance(node, ast.Assign):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Call):
                for kw in sub.keywords:
                    if kw.arg == "dtype" and isinstance(kw.value, ast.Attribute):
                        dtype = f"torch.{kw.value.attr}"
                if dtype is None and isinstance(sub.func, ast.Attribute):
                    if sub.func.attr in ("randint", "randperm"):
                        dtype = "torch.int64"
                    elif sub.func.attr in ("randn", "rand", "zeros", "ones", "empty"):
                        dtype = "torch.float32"
            if dtype:
                break
        if dtype:
            break

    # ---- postprocess source ------------------------------------------------
    # The candidate side of the comparison is often reshaped/permuted before
    # the compare: either as its own statement (`out = out.view(M, N)`) or
    # inline as the compare argument (`compare_tensors(out.view(M, N), ...)`).
    # Both are recorded as a callable applied to the kernel's return value.
    pp_src = None
    pp_arg = None
    pp_consts: dict[str, ast.expr] = {}
    if len(unit.compares) == 1 and len(ref_exprs) == 1:
        cand_expr = unit.compares[0][0]
        root = _root_name(cand_expr)
        expr = None
        if root is not None and root in defs:
            if isinstance(cand_expr, ast.Name):
                if root in postprocess:
                    expr = postprocess[root][1]
            else:
                expr = cand_expr
        if expr is not None:
            for name in _loads(expr) - {root}:
                if name not in defs:
                    raise Manual(f"postprocess uses unknown {name}")
                pp_consts[name] = _eval_const(name, defs, body, set())
            pp_src = ast.unparse(expr)
            pp_arg = root

    return {
        "make_body": make_lines,
        "ref_body": ref_lines,
        "expected": expected,
        "tol": tol,
        "dtype": dtype,
        "inplace": unit.inplace_arg,
        "pp": pp_src,
        "pp_arg": pp_arg,
        "pp_consts": pp_consts,
    }


HEADER = '''"""Specs generated from the retired test.py.

AUTO-GENERATED by tools/import_specs.py -- DO NOT EDIT BY HAND.

Each Spec is a mechanical translation of one legacy test function: the same
input construction, the same torch reference computation, and the same
tolerance the old comparison used (never widened).  Kernels the translator
refused to rewrite live in harness/specs_manual.py; the reason is printed by
the generator and recorded in the commit that introduced this file.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .registry import register
from .spec import Spec


def _shape_of(inputs):
    """Report the runtime shape of every tensor argument."""
    parts = []
    for arg in inputs[0]:
        if torch.is_tensor(arg):
            parts.append("x".join(str(d) for d in tuple(arg.shape)) or "scalar")
    return " ".join(parts) or "-"

'''

SHAPE_FN = '''

def _shape_{kernel}(inputs):
    return _shape_of(inputs)
'''


def emit(unit: Unit, info: dict) -> str:
    k = unit.kernel
    out = [f"\n\n# --- {k} (from test.py:{unit.test_fn}) ---"]
    for name, value in info["pp_consts"].items():
        out.append(f"_PP_{k.upper()}_{name} = {value!r}")
    out.append(f"\n\ndef _make_{k}(seed):")
    out.append(_indent("\n".join(info["make_body"])))
    out.append(f"\n\ndef _ref_{k}(*args):")
    out.append(_indent("\n".join(info["ref_body"])))
    if info["pp"] is not None:
        expr = info["pp"]
        if info["pp_consts"]:
            # Word-boundary match, not str.replace: replacing "M" first would
            # also rewrite the "M" inside the "N" constant that was just
            # substituted (_PP_GE_PP_GEMM_M_PP_GEMM_M_N).
            pat = re.compile(r"\b(?:" + "|".join(
                re.escape(n) for n in sorted(info["pp_consts"], key=len,
                                             reverse=True)) + r")\b")
            expr = pat.sub(lambda m: f"_PP_{k.upper()}_{m.group(0)}", expr)
        # the legacy expression still refers to the old variable name, so the
        # parameter has to carry that name (gaussian_blur uses "out")
        out.append(f"\n\ndef _pp_{k}({info['pp_arg']}):")
        out.append(_indent(f"return {expr}"))
    out.append(SHAPE_FN.format(kernel=k))
    kw = [f"name={k!r}", f"make_inputs=_make_{k}", f"reference=_ref_{k}",
          f"test_fn={unit.test_fn!r}"]
    if info["dtype"]:
        kw.append(f"dtype={info['dtype']}")
    kw.append(f"tol={info['tol']!r}")
    kw.append(f"shape=_shape_{k}")
    if info["inplace"] is not None:
        kw.append(f"inplace_indices=({info['inplace']},)")
    if info["pp"] is not None:
        kw.append(f"postprocess=_pp_{k}")
    kw.append(f'notes="translated from test.py:{unit.test_fn}"')
    out.append("\n\nregister(Spec(" + ",\n               ".join(kw) + "))")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="test.py")
    ap.add_argument("--out", default="harness/specs_legacy.py")
    ap.add_argument("--report", default="/tmp/specs_report.md")
    args = ap.parse_args(argv)

    src = Path(args.test).read_text()
    tree = ast.parse(src)
    imports = _module_imports(tree)
    module_names = set(imports) | {"torch", "math", "F"}
    module_helpers = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    ok: list[tuple[Unit, dict]] = []
    manual: list[tuple[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith("_"):
            continue
        try:
            unit = analyse(node, module_helpers)
        except Manual as exc:
            if "no leetgpu call" in str(exc):
                continue
            manual.append((node.name, str(exc)))
            continue
        if unit.kernel in EXCLUDE or node.name in EXCLUDE:
            manual.append((node.name, "excluded (hand-written spec or vacuous legacy test)"))
            continue
        try:
            info = translate(unit, module_names)
        except Manual as exc:
            manual.append((node.name, str(exc)))
            continue
        ok.append((unit, info))

    ok.sort(key=lambda ui: ui[0].kernel)
    used_imports = {imports[n] for n in ("torch", "F", "math") if n in imports}
    body = HEADER.replace(
        "import math\n\nimport torch\nimport torch.nn.functional as F\n",
        "\n".join(sorted(used_imports)) + "\n")
    parts = [body]
    for unit, info in ok:
        parts.append(emit(unit, info))
    parts.append("\n\n__all__ = [\n" + "".join(
        f"    {u.kernel!r},\n" for u, _ in ok) + "]\n")
    Path(args.out).write_text("".join(parts))

    lines = [f"# specs_legacy report", "",
             f"- translated: **{len(ok)}**", f"- manual: **{len(manual)}**", "",
             "| kernel | test_fn | manual reason |", "|---|---|---|"]
    for name, reason in sorted(manual):
        lines.append(f"| {name} | {name} | {reason} |")
    Path(args.report).write_text("\n".join(lines) + "\n")
    print(f"translated {len(ok)}, manual {len(manual)} -> {args.out}; report {args.report}")
    for name, reason in sorted(manual):
        print(f"  MANUAL {name}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
