# Extropic official workload verification through tsu
Workload source: local github.com/extropic-ai/codon_opt (Ising/DWC)
Target: tsu Z1 profile (deg 16, |J|<=6 sourced, |b|<=6 assumed, 269568 nodes)
Method: export Extropic folded-(beta,P) Ising edge list -> tsu analyse + gate_checks
Placement: light preflight only on small cases (codon graphs are often non-bipartite)

## codon_tiny_10aa
seq_len=10 P=10.0 beta=1.0 place=True
  Extropic DWC Ising: n_spins=31 n_edges=118
  analyse: deg=12 bipartite=False mediators_est=-1 |J|max=2.5 |b|max=2.99107 colour_blocks=4
    gate degree: PASS measured=12 limit=16 assumed=False
    gate coupling_cap: PASS measured=2.5 limit=6.0 assumed=False
    gate field_cap: PASS measured=2.991067409515381 limit=6.0 assumed=True
    gate node_budget: PASS measured=31 limit=269568 assumed=False
    gate colouring: PASS measured=None limit=distinct assumed=False
  preflight: verdict=fail path=failed mediators=0 err=placement failed: placement effort exhausted

## codon_default_prefix
seq_len=100 P=10.0 beta=1.0 place=True
  Extropic DWC Ising: n_spins=266 n_edges=875
  analyse: deg=12 bipartite=False mediators_est=-1 |J|max=2.5 |b|max=3.50028 colour_blocks=4
    gate degree: PASS measured=12 limit=16 assumed=False
    gate coupling_cap: PASS measured=2.5 limit=6.0 assumed=False
    gate field_cap: PASS measured=3.5002760887145996 limit=6.0 assumed=True
    gate node_budget: PASS measured=266 limit=269568 assumed=False
    gate colouring: PASS measured=None limit=distinct assumed=False
  preflight: verdict=fail path=failed mediators=0 err=placement failed: placement effort exhausted

## codon_spike_200aa
seq_len=200 P=10.0 beta=1.0 place=False
  Extropic DWC Ising: n_spins=481 n_edges=1460
  analyse: deg=12 bipartite=False mediators_est=-1 |J|max=2.5 |b|max=3.50028 colour_blocks=4
    gate degree: PASS measured=12 limit=16 assumed=False
    gate coupling_cap: PASS measured=2.5 limit=6.0 assumed=False
    gate field_cap: PASS measured=3.5002760887145996 limit=6.0 assumed=True
    gate node_budget: PASS measured=481 limit=269568 assumed=False
    gate colouring: PASS measured=None limit=distinct assumed=False

## codon_spike_full
seq_len=1273 P=10.0 beta=1.0 place=False
  Extropic DWC Ising: n_spins=3147 n_edges=9557
  analyse: deg=12 bipartite=False mediators_est=-1 |J|max=2.5 |b|max=4.05028 colour_blocks=4
    gate degree: PASS measured=12 limit=16 assumed=False
    gate coupling_cap: PASS measured=2.5 limit=6.0 assumed=False
    gate field_cap: PASS measured=4.050275802612305 limit=6.0 assumed=True
    gate node_budget: PASS measured=3147 limit=269568 assumed=False
    gate colouring: PASS measured=None limit=distinct assumed=False

## P-ramp gate sweep (tiny) — Extropic anneals P

## codon_tiny_P1
seq_len=10 P=1.0 beta=1.0 place=False
  Extropic DWC Ising: n_spins=31 n_edges=118
  analyse: deg=12 bipartite=False mediators_est=-1 |J|max=0.4 |b|max=0.885907 colour_blocks=4
    gate degree: PASS measured=12 limit=16 assumed=False
    gate coupling_cap: PASS measured=0.40000003576278687 limit=6.0 assumed=False
    gate field_cap: PASS measured=0.8859074711799622 limit=6.0 assumed=True
    gate node_budget: PASS measured=31 limit=269568 assumed=False
    gate colouring: PASS measured=None limit=distinct assumed=False

## codon_tiny_P2
seq_len=10 P=2.0 beta=1.0 place=False
  Extropic DWC Ising: n_spins=31 n_edges=118
  analyse: deg=12 bipartite=False mediators_est=-1 |J|max=0.5 |b|max=0.991067 colour_blocks=4
    gate degree: PASS measured=12 limit=16 assumed=False
    gate coupling_cap: PASS measured=0.5 limit=6.0 assumed=False
    gate field_cap: PASS measured=0.9910672903060913 limit=6.0 assumed=True
    gate node_budget: PASS measured=31 limit=269568 assumed=False
    gate colouring: PASS measured=None limit=distinct assumed=False

## codon_tiny_P5
seq_len=10 P=5.0 beta=1.0 place=False
  Extropic DWC Ising: n_spins=31 n_edges=118
  analyse: deg=12 bipartite=False mediators_est=-1 |J|max=1.25 |b|max=1.74107 colour_blocks=4
    gate degree: PASS measured=12 limit=16 assumed=False
    gate coupling_cap: PASS measured=1.25 limit=6.0 assumed=False
    gate field_cap: PASS measured=1.7410672903060913 limit=6.0 assumed=True
    gate node_budget: PASS measured=31 limit=269568 assumed=False
    gate colouring: PASS measured=None limit=distinct assumed=False

## codon_tiny_P10
seq_len=10 P=10.0 beta=1.0 place=False
  Extropic DWC Ising: n_spins=31 n_edges=118
  analyse: deg=12 bipartite=False mediators_est=-1 |J|max=2.5 |b|max=2.99107 colour_blocks=4
    gate degree: PASS measured=12 limit=16 assumed=False
    gate coupling_cap: PASS measured=2.5 limit=6.0 assumed=False
    gate field_cap: PASS measured=2.991067409515381 limit=6.0 assumed=True
    gate node_budget: PASS measured=31 limit=269568 assumed=False
    gate colouring: PASS measured=None limit=distinct assumed=False

## codon_tiny_P20
seq_len=10 P=20.0 beta=1.0 place=False
  Extropic DWC Ising: n_spins=31 n_edges=118
  analyse: deg=12 bipartite=False mediators_est=-1 |J|max=5 |b|max=5.49107 colour_blocks=4
    gate degree: PASS measured=12 limit=16 assumed=False
    gate coupling_cap: PASS measured=5.0 limit=6.0 assumed=False
    gate field_cap: PASS measured=5.491067409515381 limit=6.0 assumed=True
    gate node_budget: PASS measured=31 limit=269568 assumed=False
    gate colouring: PASS measured=None limit=distinct assumed=False

## codon_tiny_P40
seq_len=10 P=40.0 beta=1.0 place=False
  Extropic DWC Ising: n_spins=31 n_edges=118
  analyse: deg=12 bipartite=False mediators_est=-1 |J|max=10 |b|max=10.4911 colour_blocks=4
    gate degree: PASS measured=12 limit=16 assumed=False
    gate coupling_cap: FAIL measured=10.0 limit=6.0 assumed=False
    gate field_cap: FAIL measured=10.491067886352539 limit=6.0 assumed=True
    gate node_budget: PASS measured=31 limit=269568 assumed=False
    gate colouring: PASS measured=None limit=distinct assumed=False

## THRML docs 5-spin Ising chain
  deg=2 bip=True verdict=ok path=grid_embed

## Interpretation for Extropic PR
1. Official codon DWC graphs are the right verification target (paper arXiv:2606.17327).
2. gate_checks tells whether Extropic's own exported Ising violates Z1 caps at a given P.
3. If colour_blocks>2 / not bipartite: Extropic uses 4-colour Gibbs; Z1 wants bipartite chromatic — mediators/placement are tsu's job.
4. High P (constraint ramp) can push |J|=P/4 or |b| over caps — tsu should name that before silicon.
5. This is NOT an energy-equivalence proof of DWC (that's Extropic's); it is a hardware-fit audit of their published workload.

## Mediation follow-up (2026-09-13)

- tiny: 22 mediators, 53 spins, bipartite, place OK
- prefix: 168 mediators, 434 spins, gates PASS
- full spike: 1878 mediators, 5025 spins, bipartite, all Z1 gates PASS
See EXTROPIC_VERIFICATION.md for PR-ready writeup.
