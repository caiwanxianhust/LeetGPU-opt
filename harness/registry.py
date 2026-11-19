"""Spec registry: name -> Spec, with legacy test.py-name aliasing.

Specs live in three modules and all three register themselves on import:
`kernels` (hand-written, with cost models), `specs_legacy` (generated from the
retired test.py) and `specs_manual` (hand-written where the generator refused).
"""

from __future__ import annotations

from .spec import Spec

_SPECS: dict[str, Spec] = {}


def register(spec: Spec) -> Spec:
    previous = _SPECS.get(spec.name)
    if previous is not None and previous is not spec:
        raise ValueError(f"duplicate spec for {spec.name!r}")
    _SPECS[spec.name] = spec
    return spec


def load_all() -> dict[str, Spec]:
    if not _SPECS:
        # import side effects register the specs; order is irrelevant because a
        # duplicate name raises in register().
        from . import kernels, specs_legacy, specs_manual  # noqa: F401
    return _SPECS


def names() -> list[str]:
    return sorted(load_all())


def resolve(name: str) -> tuple[str, Spec]:
    specs = load_all()
    if name in specs:
        return name, specs[name]
    for canonical, spec in specs.items():
        if spec.test_fn == name:
            return canonical, spec
    raise KeyError(name)
