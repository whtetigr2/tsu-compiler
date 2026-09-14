"""R17 -- ACTUALLY TRY reproducibility, rather than reasoning about whether
it should work. Two independent tests, both against the SAME real receipt
(demo/receipts/small) this whole audit has used throughout:

  TEST A: call tsu_compiler.simulate.simulate(..., seed=999, clamp={"g0_0": 0}) with
  IDENTICAL arguments twice IN THE SAME PROCESS and diff the raw returned
  array + every field of the written simulation.json except the wall-clock
  timing field (which is expected to differ run to run and carries no
  scientific content).

  TEST B: the more realistic test -- spawn TWO SEPARATE, FRESH Python
  processes (a real user re-running the app tomorrow gets a fresh process,
  not a warm one), each calling the SAME simulate(...) with the SAME seed,
  each writing its own JSON result file, then diff those two files from
  THIS (a third) process. This is the test that would catch a reproducibility
  break caused by process-level nondeterminism (hash randomization, thread
  scheduling affecting JAX's own internal dispatch order, etc.) that a
  same-process double-call could not.

Never runs `tsuc compile`. Foreground, one-shot.

Run with:
    PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" audit/r17_reproducibility_check.py
from the repo root.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parent
SRC = REPO_ROOT / "src"
DEMO = REPO_ROOT / "demo"
RECEIPT_DIR = REPO_ROOT / "demo" / "receipts" / "small"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(DEMO))

PY = sys.executable
SEED = 999
CLAMP = {"g0_0": 0}
PARAMS = dict(n_chains=6, n_samples=3, n_warmup=600, steps_per_sample=4)


def run_test_a():
    from tsu_compiler.simulate import simulate
    print("[TEST A] same-process double call, identical seed ...")
    _p1, got1, im1 = simulate(str(RECEIPT_DIR), seed=SEED, clamp=CLAMP, **PARAMS)
    doc1 = json.loads((RECEIPT_DIR / "simulation.json").read_text())
    _p2, got2, im2 = simulate(str(RECEIPT_DIR), seed=SEED, clamp=CLAMP, **PARAMS)
    doc2 = json.loads((RECEIPT_DIR / "simulation.json").read_text())

    import numpy as np
    arrays_equal = np.array_equal(got1, got2)
    weights_equal = np.array_equal(im1.weights, im2.weights)
    biases_equal = np.array_equal(im1.biases, im2.biases)

    doc1_no_time = {k: v for k, v in doc1.items() if k != "wall_time_s"}
    doc2_no_time = {k: v for k, v in doc2.items() if k != "wall_time_s"}
    docs_equal = doc1_no_time == doc2_no_time

    print(f"  raw draw array bit-identical : {arrays_equal} (shape {got1.shape})")
    print(f"  IsingModel weights identical : {weights_equal}")
    print(f"  IsingModel biases identical  : {biases_equal}")
    print(f"  simulation.json identical (excl. wall_time_s): {docs_equal}")
    if not docs_equal:
        for k in doc1_no_time:
            if doc1_no_time.get(k) != doc2_no_time.get(k):
                print(f"    DIFFERS at key {k!r}: {doc1_no_time.get(k)!r} vs {doc2_no_time.get(k)!r}")
    return {
        "arrays_equal": bool(arrays_equal),
        "weights_equal": bool(weights_equal),
        "biases_equal": bool(biases_equal),
        "docs_equal_excl_wall_time": bool(docs_equal),
        "wall_time_s_run1": doc1["wall_time_s"],
        "wall_time_s_run2": doc2["wall_time_s"],
        "task_validity": doc1["task_validity"],
        "codeword_violation_rate": doc1["codeword_violation_rate"],
    }


def run_test_b():
    print("\n[TEST B] two SEPARATE fresh processes, identical seed ...")
    worker_script = AUDIT_DIR / "_r17_worker_subprocess.py"
    out1 = AUDIT_DIR / "traces" / "r17_procA.json"
    out2 = AUDIT_DIR / "traces" / "r17_procB.json"
    for out in (out1, out2):
        cmd = [PY, str(worker_script), str(RECEIPT_DIR), str(SEED),
               json.dumps(CLAMP), json.dumps(PARAMS), str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=170)
        print(f"  subprocess -> {out.name}: returncode={r.returncode}")
        if r.returncode != 0:
            print("  STDERR:", r.stderr[-3000:])
            raise SystemExit(f"subprocess failed for {out}")
    docA = json.loads(out1.read_text())
    docB = json.loads(out2.read_text())
    keys_to_compare = [k for k in docA if k != "wall_time_s"]
    mismatches = {k: (docA[k], docB[k]) for k in keys_to_compare if docA[k] != docB[k]}
    print(f"  process A wall_time_s = {docA['wall_time_s']:.4f}s, "
          f"process B wall_time_s = {docB['wall_time_s']:.4f}s (expected to differ)")
    print(f"  every OTHER field identical across two independent processes: "
          f"{len(mismatches) == 0}")
    if mismatches:
        for k, (a, b) in mismatches.items():
            print(f"    DIFFERS at key {k!r}: procA={a!r} procB={b!r}")
    return {
        "cross_process_identical_excl_wall_time": len(mismatches) == 0,
        "mismatches": mismatches,
        "wall_time_s_procA": docA["wall_time_s"],
        "wall_time_s_procB": docB["wall_time_s"],
    }


def main():
    print(f"[R17] repo root : {REPO_ROOT}")
    print(f"[R17] receipt   : {RECEIPT_DIR}")
    print(f"[R17] seed={SEED} clamp={CLAMP} params={PARAMS}")
    result_a = run_test_a()
    result_b = run_test_b()
    report = {"test_a_same_process": result_a, "test_b_cross_process": result_b}
    out_path = AUDIT_DIR / "traces" / "r17_reproducibility_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n[R17] wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
