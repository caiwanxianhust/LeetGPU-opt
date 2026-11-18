"""LeetGPU benchmark/decision harness.

Layout:
    spec.py      Spec contract (inputs, reference, tolerance, cost model)
    registry.py  name -> Spec, with legacy test.py-name aliasing
    measure.py   CUDA-event timing, adaptive iters, robust stats
    judge.py     accept/reject logic (pure, unit-tested by `harness selftest`)
    store.py     BENCH.json records, fingerprinted to the exact .so
    report.py    BENCH.md table rendering
    kernels.py   the actual Spec definitions
    cli.py       command line entry point
"""

__all__ = ["spec", "registry", "measure", "judge", "store", "report", "kernels", "cli"]
