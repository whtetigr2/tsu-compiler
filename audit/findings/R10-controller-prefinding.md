# R10 pre-finding (controller, during Phase A) — Onsager derivation

Recorded ahead of the R-wave so it is not lost. **R10 must re-verify independently rather
than inherit this.**

## IMPORTANT — `onsager_betac`'s docstring misderives the constant its own code computes

`demo/lattice_app.py:998-1017`

**The code is correct:**

```python
return math.log(1.0 + math.sqrt(2.0)) / 2.0 / j_max
```

`ln(1+sqrt(2))/2 = 0.4406867935097715`, which is Onsager's `K_c`. Verified independently
with SymPy: `sinh(2 * ln(1+sqrt(2))/2) = 1.000000000000` to machine precision, satisfying
Onsager's condition `sinh(2*K_c) = 1`.

**The docstring is wrong:**

> *"sinh(2*Kc) = 1, so Kc = arcsinh(1) = ln(1+sqrt(2)) and betac = Kc / j_max"*

From `sinh(2*K_c) = 1` it follows that `2*K_c = arcsinh(1)`, hence
**`K_c = arcsinh(1)/2 = ln(1+sqrt(2))/2 = 0.4407`** — not `arcsinh(1) = ln(1+sqrt(2)) =
0.8814`. The docstring drops the factor of 2 that the code correctly keeps.

**Concrete failure scenario.** A maintainer or auditor reads the docstring, computes
`K_c = ln(1+sqrt(2)) = 0.8814`, sees the code return `0.4407`, concludes the code has a
factor-2 bug, and "fixes" it — introducing a **2x error in every displayed `beta/beta_c`
ratio**. The docstring actively invites the bug it does not itself contain. Documentation
that misderives its own formula is worse than no derivation, because it looks authoritative
and is checkable in the wrong direction.

**Severity: IMPORTANT.** Nothing displayed today is wrong; the trap is latent.

**Suggested fix (for Phase F, not now):** correct the docstring to
`2*K_c = arcsinh(1) = ln(1+sqrt(2))`, so `K_c = ln(1+sqrt(2))/2`. Consider a test asserting
`sinh(2 * onsager_betac(1.0)) == pytest.approx(1.0)`, which pins the identity itself and
would fail under the docstring's version.

## What went right, and is worth stating

The function computes the constant **from the formula on every call** rather than storing a
decimal, and its docstring says why: *"a copy-pasted constant is exactly the kind of
unverified figure this project's whole ethos exists to refuse."* That instinct is correct
and is why the code survived a docstring that contradicts it.

The `ONSAGER_ASSUMPTION_NOTE` caveat is also present **on screen**, not merely in a
comment — it states that Onsager is exact only for uniform coupling on an infinite 2-D
square lattice with no field, and that this model has non-uniform couplings, hidden
mediator spins, and a different graph. R10 must still judge whether an orienting estimate
should be displayed *at all*, but it is not being passed off as derived.

## Wolfram third-party check — UNAVAILABLE so far

`00_System/Credentials/secrets.local.md` contains "Wolfram Alpha" and "wolframalpha", but
my appid extraction pattern did not match the file's actual format, so no third-party call
was made. **UNAVAILABLE: appid present in the credentials file but not parsed by the
controller's pattern.** A4 owns `wolfram_query` and should locate the correct field. The
SymPy derivation above stands on its own regardless; Wolfram would be a second independent
witness, not the only one.
