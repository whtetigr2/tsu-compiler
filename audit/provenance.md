# A3 -- The Provenance Ledger

**Method.** Every numeric claim below was checked against its stated source myself,
not taken from `.superpowers/sdd/2026-09-03-lattice-pre-theme-audit/a3-source-research.md`
(hereafter "the controller's research") on faith. Primary-source text was extracted with
`pypdf` (poppler is not installed, so `Read` cannot render these PDFs) from
`C:\Users\whtet\Downloads\TSU Bridge Reference papers and material\`:

- **`billion`** = `From One to One Billion_ Torx, Thermalizers, and Z1 - Extropic.pdf`
  (11 pages, Extropic's own product/announcement page, "SUMMER 2026").
  Extraction: `pypdf.PdfReader(...).pages[i].extract_text()`, all 11 pages.
- **`thermalizers`** = `2608.01615v1.pdf`, "Thermalizing Stochastic Programs" (55 pages,
  arXiv preprint, authors include Seth Morton). Same extraction method, all 55 pages.

Every quote below is reproduced from the raw extracted text (ligature artifacts like
`�` for "ffi"/"ffe" left as extracted, not cleaned up, so the quote is verifiably what
`pypdf` actually returned) with a page number from that same extraction (`=== PAGE N
===` markers inserted per page by the extraction script). All code line numbers were
read directly from the file at HEAD (`c421a4a3`), not carried over from A2. All computed
figures (receipt contents, gate values) were read from
`demo/receipts/small/*.json` directly, the receipt under audit throughout this task
(same one A1/A2 used: 192 nodes = 128 world spins + 64 mediator spins, 576 edges,
`beta=1.0`, `verdict=COMPILED`).

**Where I disagree with the controller's research, I say so explicitly and show the
quote that changed my mind** -- this ledger corrects that document in two places
(node budget, offset rules) beyond what it flagged as open questions, in addition to
confirming the three things it told me to expect (degree, `|J|` cap, `|b|` cap).

---

## Part 1 -- Summary table

| # | Quantity | Value in code | Class | Citation |
|---|---|---|---|---|
| 1 | `degree` | 16 | **EXTROPIC-DOCUMENTED FACT** | `thermalizers` p.5 + `billion` p.7 (two independent Extropic documents) |
| 2 | `offsets` (specific 16-tuple set) | `(1,0),(2,1),(2,3),(4,1)` + rotations | **EXTROPIC-DOCUMENTED FACT** | `thermalizers` p.5, verbatim |
| 3 | `bipartite` | `True` | **EXTROPIC-DOCUMENTED FACT** | `thermalizers` p.5, verbatim |
| 4 | `schedule` (chromatic block Gibbs) | `"chromatic_block_gibbs"` | **EXTROPIC-DOCUMENTED FACT** | `thermalizers` p.5, verbatim |
| 5 | `max_abs_coupling` used as the `\|J\|` cap | 6.0 | **EXTROPIC-DOCUMENTED FACT** (code currently says `assumed`) | `thermalizers` p.22 (Fig. 12 cap-sweep axis, "6 (Z1)") |
| 6 | the SAME `max_abs_coupling` value, reused as the `\|b\|` cap | 6.0 | **PROJECT ASSUMPTION** (code currently ALSO says `assumed` -- this one is right) | no numeric `h_max`/`b_max` found anywhere in either document |
| 7 | `node_budget` | 250,000 | **EXTROPIC-DOCUMENTED FACT**, but from a DIFFERENT, less precise Extropic document than the one giving the die's exact pbit count -- see Finding P-1 | `thermalizers` p.5, verbatim ("the entire chip has ~250,000 nodes") |
| 8 | `coupling_bits` (DAC/quantisation bit-width) | 6 | **PROJECT ASSUMPTION** (correctly labelled) | no bit-width number found anywhere in either document |
| 9 | Z1 die pbits (external reference, not modelled) | -- | **EXTROPIC-DOCUMENTED FACT** | `billion` p.7, Fig. 05 caption: "269,568 pbits" |
| 10 | Z1 die coupling parameters (external reference, not modelled) | -- | **EXTROPIC-DOCUMENTED FACT** | `billion` p.7, Fig. 05 caption: "215,904 coupling parameters" |
| 11 | coupling-parameter : edge ratio (9.988:1) | -- | **UNRESOLVED** -- no conclusion attached | not addressed by either document; see Part 4 |
| 12 | Z1 sampling rate | -- (not modelled) | **EXTROPIC-DOCUMENTED FACT**, zero display sites | `billion` p.7: ">50 MHz" |
| 13 | Z1 systems (500k / 4M / 1B pbits) | -- (not modelled) | **EXTROPIC-DOCUMENTED FACT**, zero display sites | `billion` p.7-8 |
| 14 | `\|J\|max` (this program, post-mediation) | 2.846567915192133 | **MEASURED IN SIMULATION** | `demo/receipts/small/program.json`; computed live at `demo/lattice_app.py:3170` |
| 15 | `\|b\|max` (this program) | 1.6 (mean 0.617) | **MEASURED IN SIMULATION** | `demo/receipts/small/metrics.json` (`max_abs_b`) |
| 16 | `beta` (this receipt) | 1.0 | **MEASURED IN SIMULATION** (compile-time, frozen) | `demo/receipts/small/program.json` |
| 17 | Onsager `beta_c` formula/constant | `ln(1+sqrt(2))/2` | **DERIVED FROM DOCUMENTED FACTS** (Onsager 1944; verified independently by A2 via SymPy, re-confirmed here) | classical statistical mechanics, not an Extropic document |
| 18 | `beta/beta_c` ratio (this receipt) | 6.459 (mediator-inclusive) / 2.269 (workload-only) | **DERIVED**, sensitive to an unstated modelling choice | A2 Finding I-1; re-confirmed here |
| 19 | `tau` (live ACF-plot mark) | varies per session | **EMPIRICAL OBSERVATION**, but methodologically invalid (A2 Finding C-1) | `demo/lattice_app.py:1538-1539` |
| 20 | `ess` (receipt, frozen) | `"unavailable: N/tau=1147..."` | **MEASURED IN SIMULATION**, honestly reported unavailable | `demo/receipts/small/verification.json` |
| 21 | `task_validity` | 0.24703125 | **MEASURED IN SIMULATION** (compile-time, frozen) | `demo/receipts/small/verification.json` |
| 22 | `codeword_violation_rate` | 0.01328125 | **MEASURED IN SIMULATION** (compile-time, frozen) | `demo/receipts/small/verification.json` |
| 23 | live "valid fraction" | session-dependent (e.g. 27.8% in A1's clamped trace) | **MEASURED IN SIMULATION** (live, per-session) | `demo/lattice_app.py:3506-3511` |
| 24 | mediator count (this receipt) | 64 | **MEASURED IN SIMULATION** (an artifact of the compiler's own domain-wall encoding + routing passes, NOT a hardware fact) | `demo/receipts/small/program.json` (`mediator_nodes`, 64 entries) |
| 25 | `n_nodes` / `n_edges` (this receipt) | 192 / 576 | **MEASURED IN SIMULATION** | `demo/receipts/small/metrics.json` |
| 26 | compile-pass timings (`place=201.351s` etc.) | see Part 3 | **MEASURED IN SIMULATION** (this machine, this run -- explicitly NOT a hardware timing claim) | `demo/receipts/small/passes.json` |
| 27 | `precision_headroom` / `coupling_utilisation` (regime.json) | 0.267 / 0.474 | **DERIVED FROM a PROJECT ASSUMPTION** (`coupling_bits=6`) | `demo/receipts/small/regime.json`; zero display sites (Finding P-4) |
| 28 | domain-wall mediator gadget coupling `A = arccosh(exp(2*beta*|J|))/(2*beta)` | formula, not a single number | **DERIVED** (mathematics of the encoding, not a hardware fact) | `src/tsu/passes/lower.py:34-36` (comment) |

---

## Part 2 -- Facts verified against primary sources, in detail

### 1-4 & 7. `degree`, `offsets`, `bipartite`, `schedule`, `node_budget` -- `src/tsu/target.py:51-65`

```python
Z1 = TargetProfile(
    name="z1",
    degree=Sourced(16, "F-14",
                   "All nodes in Z1 have connection rules (1,0),(2,1),(2,3),(4,1), "
                   "giving degree 16"),
    offsets=Sourced(_z1_offsets(), "F-14/F-17"),
    bipartite=Sourced(True, "F-18", "Since the graph is 2-colorable"),
    schedule=Sourced("chromatic_block_gibbs", "F-19",
                     "resampling every node v in V1 ... The same is then done for V2"),
    max_abs_coupling=Sourced(6.0, "assumed",
                             "project working value; NOT a sourced Extropic figure"),
    coupling_bits=Sourced(6, "assumed",
                          "project working value; NOT a sourced Extropic figure"),
    node_budget=Sourced(250_000, "F-15", "the entire chip has ~250,000 nodes"),
)
```

**Meta-finding before the individual facts: `"F-14"` through `"F-19"` are NOT citable from
anywhere in this repository.** I grepped the entire tree (`grep -rn "F-1[4-9]"`) and found
these ids used only inside `target.py` itself and test/receipt files that echo it back
(`tests/test_target.py`, `demo/receipts/*/target.json`). There is no fact registry,
`FACTS.md`, or comment anywhere resolving `F-14` to a document, page, or URL. A reader of
this codebase alone cannot verify a single one of these five claims -- they can only trust
the `quote` string, which is not itself checked against anything by any test. This is
Finding P-2 below.

**Independently, without using the "F-14" label at all, I opened both PDFs myself and
found every one of these five facts verbatim, all five in the SAME paragraph of the SAME
document** -- `thermalizers` (2608.01615v1.pdf), page 5, section "B. Hardware platform" /
the Z1-architecture subsection immediately following it:

> *"Nodes in the Z1 chip are arranged on a 2D grid (each node has two integer-valued
> coordinates). A node at position (x, y) with connection rule (a, b) is connected to the
> four nodes at (x+a, y+b), (x-b, y+a), (x-a, y-b), (x+b, y-a). **All nodes in Z1 have
> connection rules (1,0), (2,1), (2,3), (4,1), giving degree 16** (except at grid
> boundaries where some edges fall outside the grid)."* -- p.5

> *"The Z1 topology is a sparse, locally connected, 2-colorable graph... Since **the graph
> is 2-colorable**, there exist sets V1 and V2 such that V1 ∪ V2 = V, V1 ∩ V2 = ∅, and for
> all {u, v} ∈ E we have \|{u,v} ∩ V1\| = \|{u,v} ∩ V2\| = 1."* -- p.5

> *"The chip performs block Gibbs sampling [24-26] by first **resampling every node
> v∈V1**, setting sv=1 with probability pv(s) and sv=-1 otherwise... **The same is then
> done for V2**. These two block updates constitute one Gibbs iteration."* -- p.5

> *"The Z1 chip will perform Gibbs sampling extremely rapidly and cheaply: **the entire
> chip has ~250,000 nodes**, each Gibbs iteration is estimated to cost approximately
> 3×10⁻¹⁰ J..."* -- p.5

Independently, `billion` (the Extropic-authored product page/announcement, p.7) confirms
degree 16 a second, separate way: *"each pbit is directly connected to sixteen of its
neighbors"* and, in the Fig. 05 caption, *"The Z1 die: eight cores, 269,568 pbits,
215,904 coupling parameters, >50 MHz sampling rate, <1 W."*

**Reclassification: `degree`, `offsets` (the specific 16-tuple set, not merely the
`\|dx\|+\|dy\|` odd parity rule), `bipartite`, and `schedule` all move from "sourced only
to an uncitable internal id" to `EXTROPIC-DOCUMENTED FACT`, with a page-precise citation
each**, on the strength of my own independent verification, not the code's own `quote`
strings (which I only checked *after* finding these passages, and which turned out to
match verbatim -- the `quote` fields in `target.py` are accurate, just uncitable in-repo).

**This corrects the controller's research on one specific point.** That document said (of
the offset rules specifically): *"The specific coupling offset rules. The paper says
'following a simple pattern' and does not give the offsets."* That is true of the OTHER
paper it was reading at the time (the DTM/diffusion-models paper, which indeed does not
give offsets) -- but the Thermalizers paper does give them, verbatim, exactly matching
`target.py`'s `RULES` constant. I found this myself and it is a strictly stronger result
than what the controller's research reported: not "derived to satisfy a documented
parity rule," but directly quoted.

### 7 (continued). `node_budget = 250,000` -- the corrected finding

**The controller's research told me this was simply wrong** (*"We use 250,000. The
document says 269,568 pbits on the Z1 die... Our value matches no published figure... Not
merely conservative -- incorrect, and presented as fact."*). Per this task's own
instruction to verify rather than take that on faith, I read the Thermalizers paper myself
and found, verbatim, on page 5: *"the entire chip has ~250,000 nodes"* -- an EXACT
character-for-character match (down to the tilde) to the `quote` field already sitting in
`target.py:64`.

**So the controller's research was itself wrong here, and I am correcting it, not just
relaying it.** `node_budget=250,000` is not an uncited or fabricated figure -- it is a
verbatim quote from a real Extropic document. The controller's research had only checked
`billion` (the product-announcement PDF) and the DTM paper by the time it drew that
conclusion; it had not yet found the Thermalizers paper's own, different figure for "the
entire chip."

**But this does not fully exonerate the number either -- see Finding P-1.** Two different
Extropic-authored documents give two different figures for what both describe as "the
entire chip['s]" node/pbit count: `thermalizers` p.5 says "~250,000" (a rounded estimate,
used in that paragraph only to size a per-iteration energy-cost calculation, not presented
as a hardware spec); `billion` p.7's Fig. 05 caption says "269,568 pbits" (presented as an
exact architectural spec: *"The Z1 die: eight cores, 269,568 pbits, 215,904 coupling
parameters, >50 MHz sampling rate, <1 W"*), and the same document's own summary panel
separately states *"269,000PBITS IN A SPARSE GRAPH"*. `node_budget=250,000` matches the
OLDER, rounder, less precise of the two Extropic figures, not the newer, exact one -- both
are genuine primary sources, and I have no basis to declare one more authoritative than
the other from the documents alone (Finding P-1 states this precisely; it is a real,
unresolved cross-document discrepancy, not a defect in this project's citation).

### 5 & 6. The caps -- `src/tsu/target.py:60,62` and `src/tsu/gates.py:94-116`

```python
max_abs_coupling=Sourced(6.0, "assumed",
                         "project working value; NOT a sourced Extropic figure"),
```

I independently confirmed the controller's research finding here, from the Thermalizers
paper myself (p.22, Fig. 12, the cap-sweep figure):

> *"points indexed by cap Jmax, loose (dark) → tight (light)"* with the sweep's own axis
> values listed in the extracted text as *"0.3 0.5 0.75 1 1.5 2 3 6 (Z1) 10"* -- i.e. the
> figure explicitly labels the value **6** as **"(Z1)"**, distinguishing it from the other
> seven values swept purely as a numerical study.

And in prose, twice (p.13 and p.20):

> *"All our experiments are executed using the thrml Python library with THMs simulating
> realistic hardware by matching the topology of the Z1 thermodynamic computer and, except
> where noted, the hardware caps \|J\| ≤ Jmax, \|h\| ≤ hmax."* -- p.13

> *"every programmable weight in (39) has finite dynamic range, \|Jij\| ≤ Jmax and
> \|hi\| ≤ hmax"* -- p.20

**`Jmax=6` is Extropic's own documented Z1 coupling cap, sourced to a figure that names it
"(Z1)" explicitly** -- this reclassifies `max_abs_coupling` (as used for the `\|J\|` /
`coupling_cap` gate) from `PROJECT ASSUMPTION` to `EXTROPIC-DOCUMENTED FACT`.

**`hmax` (the `\|b\|`/field analogue) is named symbolically in both quotes above but I
found no numeric value for it anywhere in either document** -- I grepped both extracted
texts for every occurrence of "hmax" (2 hits, both symbolic, shown above) and for "6-bit",
"bit-width", "quantiz*", "DAC" (1 generic hit, no resolution number, `thermalizers` p.51:
*"E_ana,supp: The analog supporting circuitry, such as DACs, temperature sensor
circuits..."* -- names DACs exist, gives no bits). **`max_abs_coupling` used as the
`\|b\|`/`field_cap` cap therefore correctly stays `PROJECT ASSUMPTION`.**

**This is where `gates.py` becomes the crux of the whole ledger.** Read `gates.py:94-116`
(`_evaluate`):

```python
    cap = target.max_abs_coupling.value
    cap_assumed = target.is_assumed("max_abs_coupling")
    cap_source = target.max_abs_coupling.source
    ...
    j_check, j_failure = _magnitude_gate(
        "coupling_cap", "|J|max", peak_J, cap, cap_assumed, cap_source, ...)
    ...
    b_check, b_failure = _magnitude_gate(
        "field_cap", "|b|max", peak_b, cap, cap_assumed, cap_source, ...)
```

**There is only ONE `Sourced` field in `TargetProfile` for both caps** (`target.py:36`,
`max_abs_coupling: Sourced` -- there is no separate `max_abs_bias`/`h_max` field at all).
Both the `coupling_cap` gate (for `\|J\|`) and the `field_cap` gate (for `\|b\|`) read the
identical `cap`, `cap_assumed`, and `cap_source` values, because there is structurally only
one number and one `assumed` boolean to read. Confirmed live, `demo/receipts/small/gates.json`:

```json
{"gate": "coupling_cap", "measured": 2.5, "limit": 6.0, "assumed": true, "downgraded": false},
{"gate": "field_cap",    "measured": 1.6, "limit": 6.0, "assumed": true, "downgraded": false}
```

Both rows say `"assumed": true`. That was correct while both caps were genuinely unsourced.
**It is now factually wrong for `coupling_cap` and still correct for `field_cap`** -- and
because the two gates share one underlying `Sourced` object, the code cannot currently
represent the split without a schema change (a new `max_abs_bias: Sourced` field on
`TargetProfile`, threaded through `gates.py`'s two `_magnitude_gate` calls). That is a
`src/` change and out of scope for this audit (Phase A/R make no `src/` changes) -- **it is
recorded here as Finding P-3 for Phase F**, not performed.

### 14. `\|J\|max (this program)` -- a DIFFERENT number from the `\|J\|` cap, `demo/lattice_app.py:3170`

Already established by A2 (the `J`/`b` row of `truth_table.md`), re-verified here because
it is easy to conflate with entry 5 above and the conflation is itself a provenance risk:
`j_max = float(np.abs(r.im.weights).max())` is the PEAK coupling actually present in this
compiled, post-mediation program -- `2.846567915192133` per
`demo/receipts/small/program.json`, independently re-read here. This is **not** a hardware
constant, has no Extropic citation, and should never be confused with entry 5's `6.0`
hardware cap -- they are different quantities that happen to share the symbol "j_max" in
different parts of the codebase (`demo/lattice_app.py:3170`'s local variable vs.
`target.py`'s `max_abs_coupling.value`). `MEASURED IN SIMULATION` is the correct class:
it is a property of THIS compiled model, on THIS run of the compiler, not an assumption or
a hardware fact.

---

## Part 3 -- Display sites, and whether provenance is visible there

For every number above with `demo/`-side display code, every site is listed. "Provenance
visible" means a viewer looking at THAT specific spot on screen can tell, without cross-
referencing another panel, which of the six classes the number belongs to.

### `degree` (16), `bipartite`/offsets

| Site | file:line | Provenance visible? |
|---|---|---|
| VERIFICATION panel, `DEGREE` gate row | `demo/lattice_app.py:2974-3002` (loop), reads `gates.json`'s `"assumed": false` | **Yes** -- `assumed=false` correctly suppresses the `"(assumed limit)"` tag (`:2979-2980`), and this is now the CORRECT rendering (it is documented), even though the code arrived at "not assumed" only because `target.is_assumed("degree")` checks `source=="assumed"` literally and `"F-14"` happens not to equal that string -- it is right for an accidental reason, not because anything in the UI cites `thermalizers` p.5. |
| FRONTIER panel, DEGREE gauge | `demo/lattice_app.py:585-590` (`FRONTIER_GAUGE_LABELS`), `demo/frontier.py:151-154` | **No** -- the gauge shows `9 of 16` with ticks and a predicted hairline, but no citation string anywhere in `GaugeSpec`/`foot_text` for the `16` limit itself (`ASSUMED_CAP_NOTE` is appended only for `coupling_cap`/`field_cap`, `lattice_app.py:753-754` -- `degree` gets no equivalent "this is a documented Z1 fact, source: ..." note). A viewer sees an unlabelled `16` and must trust it is real without being told why. |
| explainer.py, section 05 | `demo/explainer.py:305-319` | **No** -- describes DEGREE qualitatively ("how many direct couplings any one p-bit has") without citing a source for the number 16 itself. |
| `bipartite_after` per-overlay | `demo/lattice_app.py:3800-3817` | **Yes, and correctly honest** -- already fixed per the in-code comment ("Minor #4, final review"): renders `fmt_value(self.overlay_receipt.bipartite_after)` rather than asserting bipartite as fact when that field is `None` (unrecorded) for a given receipt. No new finding here; this is the good pattern the rest of the app should match. |

### `\|J\| <= 6.0` and `\|b\| <= 6.0` -- the disclaimer that must now be split

**Every current display site renders both caps with the IDENTICAL disclaimer text**,
independently re-typed (not a single shared constant) in **five** places across **three**
files:

| # | Site | file:line | Text |
|---|---|---|---|
| a | `FOOTER_TEXT` (global app footer) | `demo/lattice_app.py:125-127` | *"...\|J\| and \|b\| caps are assumed project values, not sourced Extropic figures."* |
| b | `ASSUMED_CAP_NOTE` (FRONTIER gauge foot text, both `coupling_cap` and `field_cap`) | `demo/lattice_app.py:595-596`, appended at `:753-754` | *"assumed project limit -- \|J\| and \|b\| caps are assumed project values, not sourced Extropic figures"* |
| c | explainer.py section 05 | `demo/explainer.py:317-319` | *"\|J\| <= 6.0 and \|b\| <= 6.0 specifically are ASSUMED project values, not sourced Extropic figures -- every place this app displays them says so."* |
| d | `frontier.py` module docstring | `demo/frontier.py:36-38` | *"\|J\| <= 6.0 and \|b\| <= 6.0 ... remain ASSUMED project values, not sourced Extropic figures -- every display below says so."* |
| e | `frontier.py` `render_text` footer line | `demo/frontier.py:513-514` | *"\|J\| <= 6.0 and \|b\| <= 6.0 are ASSUMED project values, not sourced Extropic figures (spec section 4.9 / project note)."* |
| f | VERIFICATION panel gate rows, both `coupling_cap` and `field_cap` | `demo/lattice_app.py:2979-2983`, reading `gates.json`'s `"assumed": true` for BOTH | renders `"(assumed limit)"` identically for both rows |

**All six are now wrong for `\|J\|` and correct for `\|b\|`.** (c) is the most acutely
wrong: it explicitly claims universality ("every place this app displays them says so"),
which was true and is now a false claim about the app's OWN consistency, on top of the
underlying provenance error. See Finding P-3 (the fix itself, deferred to Phase F) and
Finding P-2/context above (why the fix needs a schema change, not a string edit).

### `node_budget` (250,000)

| Site | file:line | Provenance visible? |
|---|---|---|
| FRONTIER panel, NODE BUDGET gauge | `demo/lattice_app.py:589`, `755-762` | **No** -- foot text is either `"p-bits are not the constraint here"`, a bare percentage, or an unavailable note; nowhere does it say "documented, but from a different Extropic document than the die's exact spec" (Finding P-1). |
| VERIFICATION panel, `node_budget` gate row | `demo/lattice_app.py:2974-3002`, `assumed: false` in `gates.json` | **Misleadingly confident** -- `assumed=false` suppresses the `"(assumed limit)"` tag, rendering `250000` with the same unqualified confidence as `degree`'s `16`, even though (unlike degree, corroborated by TWO documents agreeing) node_budget rests on the LESS precise of two disagreeing Extropic figures. The binary `assumed`/`not-assumed` tag has no room to express "documented, but contested by another source." |
| explainer.py section 05 | `demo/explainer.py:309` | Names NODE BUDGET qualitatively only; no number, no citation. |

### Numbers documented by Extropic but not modelled anywhere (sampling rate, systems scale)

`>50 MHz` sampling rate and the 500k/4M/1B pbit system tiers (`billion` p.7-8) have **zero
display sites** -- `grep -n "MHz\|sampling rate" demo/*.py` returns nothing. `FOOTER_TEXT`
disclaims this by omission/negation (*"No hardware speed or energy claims"*,
`lattice_app.py:126`) rather than displaying and citing the real figure. Not a defect --
correctly staying silent on a quantity the app makes no claim about -- but recorded here
per Step 1's instruction to enumerate every claim the app "displays OR the compiler
asserts," including the negative case of a documented fact the app deliberately does not
surface.

### `precision_headroom` / `coupling_utilisation` (`regime.json`)

`demo/lattice_app.py:1123` loads `self.regime = load("regime.json")` (confirmed the only
occurrence of `self.regime` in the file, via `grep -n "self\.regime\b" demo/lattice_app.py`)
and **the loaded dict is never read again anywhere else in the file.** `regime.json`'s own
content for this receipt: `{"regime": "precision_limited", "basis": "two distinct |J|
values are closer together than one quantisation step and may become indistinguishable",
"coupling_utilisation": 0.4744279858653555, "precision_headroom": 0.2666666666666666,
...}` -- a compile-time verdict that this model IS in a `precision_limited` regime,
DERIVED from the assumed `coupling_bits=6`, sits fully computed in the receipt and is
never shown to a viewer. See Finding P-4.

---

## Part 4 -- The unresolved coupling-parameter ratio (explicitly parked, not settled)

Per this task's instruction, I did **not** resurrect the tied-coupling hypothesis the
controller's research already tested and disproved. I did, however, check the one lead its
research flagged as unexplored: *"Appendix J 2 is cited repeatedly for the placement
argument and the connectivity floor -- read it before finalising R8."*

I read it (extracted text, `thermalizers` pp. 46-51, Appendix J). **It does not address the
coupling-parameter count at all.** Appendix J 2 (referenced at p.20/p.24) derives the
placement/embedding argument (*"leaving an irreducible per-site residual that persists at
any value of the cap"*) and Appendix J 3 covers the meta-EBM's marginal-error measurement
definitions -- neither mentions a coupling-parameter count, a per-core budget, or anything
resembling "215,904." I grepped the full 55-page extraction for the literal strings
`"215,904"`, `"269,568"`, and `"coupling parameter"`: **zero matches anywhere in
`thermalizers`.** Those numbers exist only in `billion`'s Fig. 05 caption, which states
them without derivation.

**Status stays exactly as the controller's research left it: UNRESOLVED, no conclusion
attached.** The 9.988:1 ratio (`269,568 × 16 / 2 = 2,156,544` edges vs `215,904` coupling
parameters) is arithmetically exact enough to look structural, but neither primary source
available to this project explains it, and this compiler makes no claim that would be
falsified either way (it does not model coupling parameters as a resource at all -- see
row 27, `precision_headroom`, which is the closest existing gate, and it concerns
quantisation, not parameter *sharing*). Parked for R9, per plan.

---

## Findings

### Important

**P-1: `node_budget=250,000` is a real Extropic quote, but from a different, less precise
document than the one carrying the die's exact pbit count, and the discrepancy is invisible
everywhere the number is displayed.** `src/tsu/target.py:64` cites the value to `"F-15"`
with quote *"the entire chip has ~250,000 nodes"* -- I verified this is a verbatim,
character-for-character match to `thermalizers` p.5. Separately, `billion` p.7's Fig. 05
caption states *"The Z1 die: eight cores, 269,568 pbits, 215,904 coupling parameters..."*
for what is described, in both documents, as "the entire chip." **Concrete failure
scenario:** a viewer of the FRONTIER panel's NODE BUDGET gauge (`demo/lattice_app.py:589`,
`755-762`) or the VERIFICATION panel's `node_budget` row (`:2974-3002`, `assumed=false`,
unqualified) sees `250000` rendered with the same confidence as `degree`'s `16` -- but
`degree=16` is corroborated by TWO independent Extropic documents agreeing exactly, while
`node_budget=250,000` is contradicted by a SECOND Extropic document's more precise figure
(269,568, a ~7.8% difference). Nothing in the code or on screen distinguishes
"documented and corroborated" from "documented but contested by a more precise
figure elsewhere" -- both currently render identically (`assumed=false`, no tag). **What
was checked:** grepped `billion` and `thermalizers` extractions for "250,000", "269,568",
"269,000", "500,000", "4,000,000", "1,000,000,000" and confirmed which document states
which figure and in what context (spec caption vs. cost-estimate aside); read
`demo/receipts/small/gates.json` and `target.json` directly; confirmed `frontier.py`,
`lattice_app.py`, and `explainer.py`'s node_budget display code has no branch that could
even express a "contested" state. **Not performed (Phase F candidate):** deciding which
figure `node_budget` should track is a project/scientific decision outside this audit's
no-fix mandate; the finding is that the discrepancy exists and is currently invisible, not
which number is correct.

**P-2: The `"F-14"`-style fact ids that back five `TargetProfile` fields are not citable
from anywhere in this repository, even though (as verified independently in Part 2) their
content is accurate.** `grep -rn "F-1[4-9]"` across the whole tree returns only
`src/tsu/target.py` itself and files that echo its output (`tests/test_target.py`,
`demo/receipts/*/target.json`) -- there is no `FACTS.md`, registry, comment block, or
docstring anywhere resolving `"F-14"` to a document, page, or URL. **Concrete failure
scenario:** a future engineer (or auditor, before this task) reading `target.py:53-64`
sees five claims each stamped with a professional-looking fact id and a quote, and has no
way to independently confirm any of them without doing exactly the primary-source PDF
research this task required -- the id LOOKS like a citation and functions like a claim of
authority, but resolves to nothing checkable in-repo. This is precisely the failure mode
the `Sourced` dataclass's own module docstring says it exists to prevent ("A target profile
that silently presents an assumption as a fact is the failure the `source` field exists to
prevent") -- an uncitable id is a milder version of the same problem: not a false fact, but
an unverifiable one dressed as verified. **What was checked:** full-repo grep for the
pattern; grep for "forensic", "fact id", "fact_id", "F-14" in both this repo and the
adjacent `SPR` documentation tree (no registry found there either -- the "TSU_Knowledgebase"
hits that turned up are an unrelated project). **Silver lining, stated plainly:** every one
of the five ids I checked (F-14, F-14/F-17, F-15, F-18, F-19) turned out to resolve to a
real, accurately-quoted passage once I found the right document -- this is a citability gap,
not a fabrication. `test_target.py:16-20`'s own test name
(`test_coupling_cap_is_marked_assumed_not_sourced`) and docstring (*"Jmax is this project's
working value, NOT an Extropic figure"*) is now factually superseded by the `thermalizers`
p.22 cap-sweep finding -- flagged here as a second, smaller instance of the same
"prior tests are evidence, not immunity" pattern the plan warns about; not fixed, per the
no-repair rule.

**P-3: The `\|J\|`/`\|b\|` disclaimer is asymmetrically wrong, in five independently-typed
locations, and cannot currently be split without a `TargetProfile` schema change.** See
Part 3 above for the full table of five sites (`lattice_app.py:125-127`, `:595-596`+`:753-
754`, `explainer.py:317-319`, `frontier.py:36-38`, `frontier.py:513-514`) plus the
VERIFICATION panel's `"(assumed limit)"` tag (`lattice_app.py:2979-2983`, sourced from
`gates.py:94-116`'s shared `cap`/`cap_assumed` values). **Concrete failure scenario:** a
viewer reading explainer.py's section 05 (*"...every place this app displays them says
so"*) has been told, in-app, that a single disclaimer note reliably describes BOTH caps
everywhere -- that claim about the app's own consistency was true when both caps were
equally unsourced and is now false in the specific direction of UNDERCLAIMING `\|J\|`'s
provenance (disclaiming a fact as an assumption), which the plan calls out as the newly-
important failure direction, the converse of every previous overclaiming finding in this
project. **Why this cannot be a one-string fix:** `src/tsu/target.py:36-38` has exactly one
`Sourced` field, `max_abs_coupling`, feeding BOTH `gates.py:104`'s `coupling_cap` check and
`gates.py:112`'s `field_cap` check (`gates.py:94-96`, `cap`/`cap_assumed`/`cap_source` are
each read once and reused for both `_magnitude_gate` calls) -- splitting the disclaimer
requires a new `max_abs_bias: Sourced` field on `TargetProfile`, a second read site in
`gates.py`, and five text edits in `demo/`, all of which are `src/`-touching changes
forbidden during Phase A/R. **What was checked:** read `target.py`, `gates.py` in full;
confirmed via `demo/receipts/small/gates.json` that both `coupling_cap` and `field_cap`
rows currently carry identical `"limit": 6.0, "assumed": true`; confirmed each of the five
display-site strings is independently typed (not one shared constant reused across files --
`ASSUMED_CAP_NOTE` IS shared within `lattice_app.py` itself, but `frontier.py` and
`explainer.py` each have their own independently-typed copy of the same sentence, meaning a
partial fix that only edits `ASSUMED_CAP_NOTE` would still leave three sites wrong).
**Recorded for Phase F, not performed here** (schema change + five text edits, in that
order, with a failing test for each before the fix, per the plan's TDD requirement for
Phase F).

### Minor

**P-4: `regime.json`'s `precision_limited` verdict is computed at every compile and never
shown to a viewer.** `demo/lattice_app.py:1123` (`self.regime = load("regime.json")`) is
the ONLY occurrence of `self.regime` in the file (confirmed by grep) -- the loaded dict,
which for this receipt reads `{"regime": "precision_limited", "coupling_utilisation":
0.474, "precision_headroom": 0.267, "basis": "two distinct \|J\| values are closer together
than one quantisation step and may become indistinguishable", ...}`, is loaded and then
never read again. **Concrete failure scenario:** none directly (nothing false is displayed,
because nothing is displayed) -- but the compiler has already determined, honestly and
correctly, that this specific model sits in a regime where two of its own couplings could
become numerically indistinguishable under the assumed 6-bit DAC resolution, and that
finding is silently discarded rather than shown, disclaimed as resting on `coupling_bits`
(itself a `PROJECT ASSUMPTION`, entry 8), or surfaced anywhere a viewer could see it. A
viewer has no way to know this analysis exists, let alone that it flagged a concern.
**What was checked:** grep confirmed the single load site and no re-read; read `regime.py`
in full to confirm `precision_limited`/`coupling_utilisation`/`precision_headroom` are
genuinely computed (not placeholder/always-same values -- `regime.py`'s own module
docstring documents a PRIOR bug of exactly that shape, `precision_headroom` being a
disguised constant, already fixed per that same docstring, so this is not a re-discovery of
that old bug, just a note that the now-correct computed value has no display path at all).

### What was tried and did not turn up a finding

- Checked whether `coupling_bits=6` ("assumed") has any citable source the way `degree`
  and the caps did -- grepped both extractions for "bit-width", "bit width", "quantiz",
  "6-bit", "6 bit", "DAC". One generic hit (`thermalizers` p.51, names DACs exist as
  supporting circuitry, gives no resolution number). No finding: the code's own `"assumed"`
  label for `coupling_bits` is correct as-is, and I found nothing to reclassify it against.
- Checked whether the `\|b\|` cap might ALSO be documented somewhere I hadn't looked, since
  `\|J\|`'s cap turned out to be citable in a place the controller's research had not yet
  found (a cap-sweep figure, not prose) -- grepped every occurrence of "hmax" (2 hits, both
  symbolic definitions, `thermalizers` pp.13,20) and every occurrence of a bare number near
  "field" or "bias" or "h_max" in both extractions. No numeric value for `hmax` anywhere.
  No finding: `PROJECT ASSUMPTION` for `\|b\|` stands, confirmed rather than merely
  inherited.
- Checked whether `bipartite_after` being asserted as fact anywhere else in the codebase
  (beyond the already-fixed site A2's truth table and this task both read at
  `lattice_app.py:3800-3817`) -- grepped every occurrence of `"bipartite"` in `demo/*.py`.
  All other occurrences are either the target-level (`Z1.bipartite`, now correctly
  EXTROPIC-DOCUMENTED) or the same already-honest `fmt_value(...bipartite_after)` pattern.
  No finding.
- Checked whether the `\|J\|`/`\|b\|` disclaimer split (P-3) is a SILENT UI bug already,
  i.e. whether any code path currently uses `cap_assumed`/`cap_source` differently for the
  two gates despite reading from the same field -- read `_magnitude_gate` and both call
  sites in `gates.py:94-116` line by line; they are structurally identical calls differing
  only in `gate`/`label`/`peak`/`encoding_hint`. No hidden asymmetry; the two gates are
  provably identical in every field that would need to differ for the split, confirming P-3
  requires a schema change and cannot be patched by a smarter read of the existing data.

---

## Part 5 -- Oracle cross-check record (Task A4 tie-in)

Recorded here because this ledger is the natural home for "how was a MEASURED number
independently verified," per the plan's own provenance framing. Full detail (TDD cycle,
watched failure, implementation) is in `audit/oracles/`; the numeric result of the A4
Step 6 cross-check is:

- **Local** (`audit/oracles/exact.py`, `exact_boltzmann`, independent of `src/tsu`): two
  spins, `J={(0,1): 1.0}`, `b=[0,0]`, `beta=1.0` -- aligned/anti-aligned probability ratio
  = **`7.3890560989306495`**.
- **Wolfram|Alpha** (`audit/oracles/wolfram.py`, `wolfram_query("N[exp(2), 12]")`, the AppID
  read in-memory from `secrets.local.md`, never persisted) -- **`'7.38905609893'`**.
- **Agreement**: match to 11 significant figures (`math.isclose(..., rel_tol=1e-4)` passes
  with enormous margin). Wolfram Alpha was reachable during this task; the query above is
  the actual live result, not a placeholder.

This is the strongest kind of confirmation available to this audit: a hand-derivable
physical quantity (`exp(2)`, from the plan's own worked example), computed by two
completely independent code paths -- one written from the physics with no import of
`src/tsu`, one evaluated by a third party's own servers -- agreeing to machine precision.
