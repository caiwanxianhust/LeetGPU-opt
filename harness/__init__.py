"""LeetGPU benchmark/decision harness.

Layout:
    spec.py      Spec contract (inputs, reference, tolerance, cost model)
    registry.py  name -> Spec, with legacy test.py-name aliasing
    measure.py   CUDA-event timing, adaptive iters, robust stats
    judge.py     accept/reject logic (pure, unit-tested by `harness selftest`)
    store.py     BENCH.json records, fingerprinted to the exact .so
    report.py    BENCH.md table rendering
    checker.py   output comparison semantics of `harness check` (pure)
    kernels.py   hand-written Specs with cost models
    specs_legacy.py  Specs generated from the retired test.py (do not edit)
    specs_manual.py  hand-written Specs for the kernels the generator skipped
    cli.py       command line entry point
"""

__all__ = ["spec", "registry", "checker", "measure", "judge", "store", "report",
           "kernels", "specs_legacy", "specs_manual", "cli"]
