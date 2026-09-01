"""A1: seam-clamp measurement -- can a chunk be sampled subject to its
neighbours' already-sampled edges, cheaply and correctly?

Everything here rides on the VERIFIED `tsu.simulate.simulate` API and the
compiler's OWN `encode()` (`is_codeword` / `decode` / `encode_clamp`) and
`spec.contract.validate` -- exactly the pattern `demo/render_world.py` uses.
No clamping or decoding is hand-rolled here; only the CROSS-CHUNK adjacency
check (water directly against rock, no grass between) is hand-rolled, because
that check spans two independently-sampled grids and the compiler's contract
only ever validates edges *within* one receipt's own grid.

A1a -- single-edge seam: does clamping chunk B's west column to chunk A's
       east column (a) get honoured exactly, (b) still find valid worlds at
       a usable rate, and (c) produce a seam that is actually legal where
       A and B would physically touch.

A1b -- two-edge corner hazard: chunk C is clamped on both its north row (from
       A) and west column (from B), where A and B were never sampled against
       each other. Two distinct, separately-counted failure modes:
         1. corner disagreement -- A and B assign C's shared corner cell
            different values; the clamp is self-contradictory before
            sampling starts.
         2. corner-consistent but infeasible -- clamp agrees, no valid C
            found.

Run: PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" demo/seam_check.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from tsu.simulate import simulate
from tsu.spec import load_spec
from tsu.passes.encode import encode

R = "demo/receipts/small"
W = H = 8
WATER, ROCK, GRASS = 0, 1, 2

spec = load_spec(str(Path(R) / "spec.yaml"))


def _selected_encoding(receipt_dir) -> str:
    passes = json.loads((Path(receipt_dir) / "passes.json").read_text())
    for c in passes["candidates"]:
        if c["state"] == "SELECTED":
            return c["encoding"]
    raise ValueError(f"receipt at {receipt_dir} has no SELECTED candidate")


enc = encode(spec, _selected_encoding(R))


def classify(got, im):
    """Split a batch of raw draws into (valid decoded worlds, non-codeword
    count, codeword-but-contract-invalid count), reusing the compiler's OWN
    is_codeword / decode / contract.validate -- the render_world.py pattern.
    """
    valid, noncodeword, invalid = [], 0, 0
    for row in got:
        bits = dict(zip(im.nodes, row.tolist()))
        if not enc.is_codeword(bits):
            noncodeword += 1
            continue
        dec = enc.decode(bits)
        if spec.contract.validate(dec).ok:
            valid.append(dec)
        else:
            invalid += 1
    return valid, noncodeword, invalid


def clamp_honoured(got, im, workload_clamp):
    """Physical-spin-level check: for EVERY returned draw (codeword or not,
    valid or not), do the clamped spins hold the exact values encode_clamp
    demands? This is the strict form of "the clamp was honoured" -- a
    violation here is a sampler/program bug, not a statistic.
    """
    encoded = enc.encode_clamp(workload_clamp)
    idx = {n: i for i, n in enumerate(im.nodes)}
    ok = 0
    bad = []
    for row in got:
        good = all(int(row[idx[name]]) == val for name, val in encoded.items())
        if good:
            ok += 1
        else:
            bad.append(row)
    return ok, len(got), bad


def find_one_valid(seed, n_chains=6, n_samples=20, n_warmup=600, clamp=None,
                    max_tries=4):
    """Sample unclamped/clamped until at least one valid decoded world turns
    up (unclamped baseline is ~25%/draw, so this essentially never needs a
    second try). This is scaffolding to obtain ONE representative valid
    neighbour state to build a clamp from -- it is not part of any measured
    rate in this report; nothing here is re-rolled to make C's OWN sampling
    look better (that would defeat the point of A1b).
    """
    for attempt in range(max_tries):
        path, got, im = simulate(R, n_chains=n_chains, n_samples=n_samples,
                                 n_warmup=n_warmup, seed=seed + attempt * 97,
                                 clamp=clamp)
        valid, _, _ = classify(got, im)
        if valid:
            return valid[0], got, im
    raise RuntimeError(
        f"no valid draw in {max_tries} attempts at seed {seed} "
        f"(clamp={clamp}) -- this would itself be a finding")


def seam_pairs_legal(left, right, left_x, right_x):
    """The 8 cross-boundary (y=0..7) cell pairs between column `left_x` of
    `left` and column `right_x` of `right`. Returns (bad_pairs, all_pairs)."""
    pairs = [(left[f"g{left_x}_{y}"], right[f"g{right_x}_{y}"]) for y in range(H)]
    bad = [p for p in pairs if set(p) == {WATER, ROCK}]
    return bad, pairs


# ---------------------------------------------------------------------------
# A1a -- single-edge seam
# ---------------------------------------------------------------------------

def a1a():
    print("=" * 70)
    print("A1a -- single-edge seam")
    print("=" * 70)

    A, gotA, imA = find_one_valid(seed=1)
    print(f"chunk A: sampled unclamped, {len(gotA)} draws, "
          f"picked 1st valid decoded world as A")

    east_col = {f"g0_{y}": A[f"g7_{y}"] for y in range(H)}
    print(f"A's east column (g7_0..g7_7): "
          f"{[A[f'g7_{y}'] for y in range(H)]}")
    print(f"clamp for B's west column (g0_0..g0_7): "
          f"{[east_col[f'g0_{y}'] for y in range(H)]}")

    n_chains, n_samples, n_warmup, seed_b = 8, 30, 600, 500
    path_b, got_b, im_b = simulate(R, n_chains=n_chains, n_samples=n_samples,
                                   n_warmup=n_warmup, seed=seed_b,
                                   clamp=east_col)
    n_draws = len(got_b)
    print(f"chunk B: sampled with west-column clamp, n_chains={n_chains}, "
          f"n_samples={n_samples}, n_warmup={n_warmup}, seed={seed_b} "
          f"-> {n_draws} draws")
    assert n_draws >= 200, f"need >=200 draws, got {n_draws}"

    # (1) clamp honoured, every draw, physical-spin level
    ok, total, bad_rows = clamp_honoured(got_b, im_b, east_col)
    print(f"\n[1] clamp honoured: {ok}/{total} draws exact "
          f"({'ALL' if ok == total else f'{len(bad_rows)} VIOLATIONS -- CRITICAL BUG'})")
    if bad_rows:
        print("    CRITICAL: a returned sample did not honour the clamp. "
              "This is a bug in the sampler/program construction, not a "
              "statistic. Stopping here.")
        return {"critical_bug": "clamp not honoured", "bad_count": len(bad_rows)}

    # (2) valid fraction under clamp
    valid_b, noncodeword_b, invalid_b = classify(got_b, im_b)
    valid_fraction = len(valid_b) / n_draws
    print(f"\n[2] B's valid-state fraction under the west-column clamp: "
          f"{len(valid_b)}/{n_draws} = {valid_fraction:.4f} "
          f"({100*valid_fraction:.1f}%) "
          f"vs ~25% unclamped baseline")
    print(f"    ({noncodeword_b} non-codewords, {invalid_b} codewords "
          f"failing the contract, out of {n_draws} draws)")
    # sanity: independently confirm the baseline off A's OWN unclamped draws
    valid_a_all, _, _ = classify(gotA, imA)
    print(f"    (sanity check -- A's own unclamped valid fraction this run: "
          f"{len(valid_a_all)}/{len(gotA)} = {len(valid_a_all)/len(gotA):.4f})")

    # (3) seam legality
    print(f"\n[3] seam legality across {len(valid_b)} valid B draws:")
    trivial_bad_total = 0
    honest_bad_total = 0
    honest_legal_draws = 0
    for Bstate in valid_b:
        bad_trivial, _ = seam_pairs_legal(A, Bstate, 7, 0)
        bad_honest, _ = seam_pairs_legal(A, Bstate, 7, 1)
        trivial_bad_total += len(bad_trivial)
        honest_bad_total += len(bad_honest)
        if not bad_honest:
            honest_legal_draws += 1

    print(f"    TRIVIAL variant (A.x=7 vs B.x=0, the clamped/duplicated "
          f"column itself): {trivial_bad_total} bad pairs across "
          f"{len(valid_b)*H} pairs checked.")
    print(f"    This is EXPECTED to be trivially legal: B.x=0 was clamped "
          f"EQUAL to A.x=7, so every pair is (v, v) and can never be the "
          f"forbidden {{water, rock}} set. A trivially-satisfied seam like "
          f"this is much weaker evidence than it looks -- it says nothing "
          f"about the seam an actual neighbouring chunk would present.")
    print(f"    HONEST variant (A.x=7 vs B.x=1, treating the clamped "
          f"column as the SHARED/overlapping boundary column rather than a "
          f"duplicate -- this is the pair that matters if A and B are laid "
          f"down side by side with column x=7 of A meaning the same "
          f"physical column as x=0 of B): {honest_bad_total} bad pairs "
          f"across {len(valid_b)*H} pairs checked "
          f"({honest_legal_draws}/{len(valid_b)} draws fully legal).")

    return {
        "clamp_honoured": (ok, total),
        "valid_fraction": valid_fraction,
        "n_draws": n_draws,
        "trivial_bad_total": trivial_bad_total,
        "honest_bad_total": honest_bad_total,
        "honest_legal_draws": (honest_legal_draws, len(valid_b)),
    }


# ---------------------------------------------------------------------------
# A1b -- two-edge corner hazard
# ---------------------------------------------------------------------------

def a1b(n_pairs=50):
    print()
    print("=" * 70)
    print("A1b -- two-edge corner hazard")
    print("=" * 70)

    disagreement = 0
    infeasible = 0
    feasible = 0
    feasible_valid_fractions = []
    pooled_valid = 0
    pooled_total = 0

    n_chains_ab, n_samples_ab, n_warmup_ab = 4, 10, 600
    n_chains_c, n_samples_c, n_warmup_c = 8, 30, 600

    for i in range(n_pairs):
        seed_a = 10_000 + i * 13
        seed_b = 50_000 + i * 13
        A, _, _ = find_one_valid(seed_a, n_chains=n_chains_ab,
                                 n_samples=n_samples_ab, n_warmup=n_warmup_ab)
        B, _, _ = find_one_valid(seed_b, n_chains=n_chains_ab,
                                 n_samples=n_samples_ab, n_warmup=n_warmup_ab)
        # A is NORTH of C: A's south row (y=7) -> C's north row (y=0)
        north_clamp = {f"g{x}_0": A[f"g{x}_7"] for x in range(W)}
        # B is WEST of C: B's east column (x=7) -> C's west column (x=0)
        west_clamp = {f"g0_{y}": B[f"g7_{y}"] for y in range(H)}

        corner_from_a = north_clamp["g0_0"]   # = A's g0_7
        corner_from_b = west_clamp["g0_0"]    # = B's g7_0

        if corner_from_a != corner_from_b:
            disagreement += 1
            print(f"  pair {i:2d}: CORNER DISAGREEMENT "
                  f"(A says {corner_from_a}, B says {corner_from_b}) "
                  f"-- clamp self-contradictory, C not sampled")
            continue

        # Merge explicitly -- corner key present in both; verified equal
        # above, so this is NOT a silent dict-literal overwrite.
        c_clamp = dict(north_clamp)
        c_clamp.update(west_clamp)
        assert len(c_clamp) == 2 * W - 1  # 15: 8 + 8 - 1 shared corner

        path_c, got_c, im_c = simulate(
            R, n_chains=n_chains_c, n_samples=n_samples_c,
            n_warmup=n_warmup_c, seed=90_000 + i * 13, clamp=c_clamp)
        valid_c, _, _ = classify(got_c, im_c)
        pooled_valid += len(valid_c)
        pooled_total += len(got_c)

        if not valid_c:
            infeasible += 1
            print(f"  pair {i:2d}: corner consistent (={corner_from_a}), "
                  f"C INFEASIBLE -- 0/{len(got_c)} valid draws")
        else:
            feasible += 1
            frac = len(valid_c) / len(got_c)
            feasible_valid_fractions.append(frac)
            print(f"  pair {i:2d}: corner consistent (={corner_from_a}), "
                  f"C feasible -- {len(valid_c)}/{len(got_c)} = {frac:.3f} valid")

    n = n_pairs
    n_consistent = n - disagreement
    disagreement_rate = disagreement / n
    infeasible_rate = infeasible / n
    infeasible_rate_of_consistent = infeasible / n_consistent if n_consistent else float("nan")
    mean_feasible_valid_fraction = (
        sum(feasible_valid_fractions) / len(feasible_valid_fractions)
        if feasible_valid_fractions else float("nan"))
    pooled_valid_fraction = pooled_valid / pooled_total if pooled_total else float("nan")

    print()
    print(f"over {n} independent (A, B) pairs:")
    print(f"  corner disagreement: {disagreement}/{n} = {disagreement_rate:.3f}")
    print(f"  corner-consistent-but-infeasible: {infeasible}/{n} = "
          f"{infeasible_rate:.3f} of ALL {n} pairs "
          f"({infeasible}/{n_consistent} = {infeasible_rate_of_consistent:.3f} "
          f"of the {n_consistent} corner-consistent pairs)")
    print(f"  feasible: {feasible}/{n}")
    print(f"    mean per-pair valid fraction when feasible: "
          f"{mean_feasible_valid_fraction:.4f}")
    print(f"    pooled valid fraction across feasible pairs' draws: "
          f"{pooled_valid}/{pooled_total} = {pooled_valid_fraction:.4f}")

    return {
        "n_pairs": n,
        "disagreement": disagreement,
        "disagreement_rate": disagreement_rate,
        "infeasible": infeasible,
        "infeasible_rate": infeasible_rate,
        "infeasible_rate_of_consistent": infeasible_rate_of_consistent,
        "feasible": feasible,
        "mean_feasible_valid_fraction": mean_feasible_valid_fraction,
        "pooled_valid_fraction": pooled_valid_fraction,
        "pooled_valid": pooled_valid,
        "pooled_total": pooled_total,
    }


if __name__ == "__main__":
    result_a1a = a1a()
    result_a1b = a1b(n_pairs=50)
    print()
    print("=" * 70)
    print("SUMMARY (also see .superpowers/sdd/.../a1-report.md)")
    print("=" * 70)
    print(json.dumps({"a1a": result_a1a, "a1b": result_a1b}, indent=2, default=str))
