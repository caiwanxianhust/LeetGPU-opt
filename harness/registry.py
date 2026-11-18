"""Spec registry: name -> Spec, with legacy test.py-name aliasing."""

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
        from . import kernels  # noqa: F401  (import side effect registers specs)
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
