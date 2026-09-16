# The Thermodynamic Workbench — design

**Date:** 2026-09-16
**Status:** design approved; implementation plan next
**Working name:** "Workbench". Not final — see Open Questions.

---

## 1. What this is

One program. Authoring, compilation, sampling, diagnostics, and rendering
for Ising machines.

It ships as a single Windows executable. It runs offline, on CPU. It needs no
account, no network, no server the user has to start, and no Extropic
hardware. You write a thermodynamic program in it, it tells you whether that
program will physically fit on the chip, it samples the program, and it shows
you what came out — as numbers, as physics, and as a picture if the program
has one.

The shape to aim at is Blender: a viewport that dominates, editors instead of
screens, a non-destructive pipeline you can click through stage by stage, and
a scripting layer that drives all of it.

**The success criterion, in the owner's words:** a thing you can say
"hey, go download my program" about. That is not a finishing touch. It is the
acceptance test, and section 17 sequences the work so it is true early and
stays true.

---

## 2. The governing rule

**No capability ships that only an agent can invoke.**

The compiler currently has one practical interface: ask an AI to write a
script against it. That means the capability exists and the product does not.
Every script in `audit/` is a missing button.

This rule is the acceptance test for every feature. A feature is not done when
the measurement is correct. It is done when a person who has never met this
repository can get that measurement by clicking something.

---

## 3. Which chassis, and why

Two applications exist today.

**LATTICE** (`demo/lattice_app.py`) is a single 5,233-line Tkinter program.
Its content is excellent — pipeline, frontier, decoded world, regime and
traces, scope, layers. Its layout code is largely scar tissue from fighting
the toolkit: panels that received two pixels instead of their natural height,
an entire row of seven panels silently unmapped below a certain window size, a
disclosure frozen at one pixel square. Tkinter cannot reach the GPU. No WebGL,
no shaders, no 3D, no smooth animation. Every pixel is drawn on the CPU
through PIL. The ceiling here is the toolkit, not the design.

**Gibbs Observatory** (`demo/gibbs-observatory/`) is React and Vite on a
FastAPI backend, with eleven working views: overview, spins, couplings,
connectivity, schedule, residuals, scope, state space, notepad, and two labs.
Being a web stack, GPU rendering is simply available.

**Decision: re-shell the Observatory.** Keep its backend and its view
components. Replace its shell. Absorb LATTICE's panels as editor types and its
world generator as a decoder. Retire `lattice_app.py` by absorption, not by
rewrite.

The reasoning that settled it: the shell is tabs and splitters, which is cheap
either way. The viewport, the sweep timeline, the interactive pass stack, and
the pluggable decoder are the real work, and *neither* application has any of
them. Rebuilding a shell from scratch buys nothing and costs the eleven views
that already work.

---

## 4. The Blender mapping

| Blender | Here |
|---|---|
| 3D viewport | The lattice in space. Spins as objects, live, orbitable while sampling runs |
| Outliner | Program structure: nodes, colour blocks, edges, clamped vs free, mediators |
| Properties, context-sensitive | Click a spin: bias, local field, P(+1), its block. Click an edge: J, and whether that limit is sourced or assumed |
| Timeline / frame scrubber | Gibbs sweeps. Scrub sampling history in both directions |
| Modifier stack | The compiler passes, each stage clickable, showing the model as it stood at that point |
| Text editor and Python console | The program editor and a live console |
| Viewport shading modes | Raw spin state / decoded object / final render |
| F12 render | The thing that popped out |

The modifier-stack mapping is the important one. Blender's contribution was not
the viewport; it was making a non-destructive pipeline legible. The compiler is
already that pipeline. Today it prints a verdict. As a stack you click through,
it becomes the thing people screenshot.

---

## 5. Workspaces

Three, in the Blender sense: named arrangements of editors, switchable from
tabs beside the menu bar. Not modes, and not three applications.

- **Build** — program editor, compiler stack, outliner. For writing.
- **Observe** — instruments. Energy, structure, diagnostics, state space.
- **Render** — viewport dominant, game full-bleed, telemetry to an overlay.

This is how "one tool, three modes" is delivered. It also answers the
presentation and teaching cases: a guided walkthrough rides on top of Build
and Observe (the Observatory already has a Walkthrough component to build on),
and presentation mode is Render with the chrome hidden.

Menu bar: File, Edit, Program, Sample, Window, Help.

---

## 6. Pane system

Menu bar, workspace tabs, and resizable split panes. Each pane hosts an editor
chosen from a dropdown, the way Blender's editor-type selector works.

**Explicitly not in v1:** arbitrary pane splitting and joining, floating
windows, user-saved custom workspaces. Three well-designed fixed workspaces
with resizable splits get most of the feel at a fraction of the cost. Revisit
only if the fixed arrangements actually bind.

---

## 7. The load, compile, run loop

```
IDLE -> LOADED -> COMPILING -> OK ----> RUNNING
                      +------> REFUSED
```

Three ways in, all equal: type into the editor, paste, or point at a file on
disk. Format is sniffed, never asked for.

**REFUSED is a designed screen, not an error state.** Anyone can print
"compile failed". This one says which gate, the measured value, the cap it
broke, whether that cap is a published Extropic figure or this project's own
assumption, and what to change. It is the single best argument for the whole
project and gets as much design attention as the success path.

### Two-tier compile

Full compilation cannot run on every keystroke. Placement takes seconds, and
one existing receipt took over three minutes.

- **Gates run live, as you type.** Degree, coupling and bias caps,
  colourability, node budget. All size-invariant and cheap. Break one and it is
  marked immediately.
- **Placement runs on demand.** The expensive search, using the existing effort
  ladder, reported with the effort it actually needed.

This is Blender's viewport preview versus F12, for the same reason: a fast
approximate answer while you work, an authoritative slow one when you commit.

---

## 8. Programs

Two kinds of input, which matters more than the file extensions suggest.

**Static models** — `.json` edge lists and `.yaml` workload specs. One lattice,
fixed numbers. `load_model` already accepts both.

**Programs** — `.py`. These build models on demand, and this is not a
convenience. The game cannot be anything else: its biases are rewritten every
frame from where the player is standing, so no single file *is* its model.
Something has to *build* the model.

A loadable program declares three things:

    build(inputs) -> model      # makes the lattice
    input_ports                 # which spins external input clamps
    decoder                     # how what pops out becomes a picture

Blender is driven by Python for the same reason, and that is a large part of
why it became what it is.

**Trust.** Loading a `.py` from disk runs someone else's code. There is no
clever way around it — the deal is the same as Blender's, or a spreadsheet
with macros. Handling: a clear consent prompt on load from disk, naming the
file and saying plainly that it contains code and will execute. No prompt for
models typed or pasted into the editor, because those are data. Do not claim a
sandbox that does not exist.

---

## 9. Input is clamping

One mechanic underlies every form of interaction in the suite: **pin some
spins, let the rest relax.**

Press a movement key and the player's cell is clamped. Grab a residue and hold
it, and that residue is clamped. Fix three amino acids in a codon design, and
they are clamped. Set a boundary condition on a world, and the edge is clamped.

This is not an analogy forced onto the problem. Extropic's own Boltzmann
wrapper names them `input_blocks` and `output_blocks`, clamped at sample time,
everything else free. The interaction model of the application is the
hardware's actual interface. The game already works this way, rewriting biases
per frame in JavaScript rather than through a named port.

---

## 10. Decoders

A decoder is a first-class plugin providing three things:

1. **Decode** — spins to object (a world grid, a sequence, a conformation, a
   game frame, a texture)
2. **Present** — the object in its own idiom (a walkable level, ball and stick,
   residue letters on a track, a material swatch)
3. **Drive** — user input to clamps

Plus one that falls out for free: **energy attributed back onto the object.**
Not only a trace of total energy, but energy painted per bond, per cell, per
residue — because energy is a sum over edges and every edge belongs to some
part of the decoded object. The global trace stays; they answer different
questions.

### The viewport already exists in prototype

Found while retiring the Alloy Lab: it and the EBM Lab used seventeen of the
same nineteen CSS classes. They were not two labs. They were **one view with two
decoders**, differing only in what turns spins into a picture:

    hero image  |  side gauges  |  sample a batch  |  thumbnail gallery

That is the decoder viewport this section describes, built already without being
named. Its classes are renamed from `alloy-*` to `lab-*` and it is kept rather
than rewritten. The Alloy Lab was removed because its *readings* were general
ones wearing a model-specific label; the EBM Lab stays because it is the working
example of the seam.

This lowers the cost of the decoder work: the presentation half exists, and what
has to be built is the contract underneath it.

### v1 decoders

- **Game** — spatial, interactive, pointer-locked
- **Sequence** — non-spatial, runs Extropic's own codon workloads
- **Sprite and material forge** — `audit/texture_from_energy.py` already does
  this; the game's materials are a handful of numbers per material with zero
  stored pixels

Two shapes that different passing through one seam is what proves the seam is
real.

### Later

- **HP lattice protein** — a chain folding on a grid, energy as contact count.
  Genuinely Ising-shaped, fits Z1's connectivity, and the literature's own name
  for it is "the Ising model of protein folding" (arXiv:1102.0308). This is the
  demo to aim at an audience.
- **World and terrain** — LATTICE's generator, absorbed.

### The line we do not cross

Molecular dynamics and real drug-discovery molecules are different physics and
a different solver. A lattice protein is to those what a wind tunnel model is
to an airplane: real, useful, and not the airplane. Labelled that way it is
impressive. Labelled as "we fold proteins" it collapses in the first five
minutes of a serious conversation.

---

## 11. The compiler pass stack

Rendered as Blender's modifier stack: an ordered list, each entry clickable,
each showing the model as it stood at that point.

Degree check, coupling and bias caps, colourability, node budget, placement,
mediator insertion, precision. Each carries its verdict, its measured value,
its cap, and the provenance of that cap.

Two views deserve their own editors:

- **Placement** — the model laid onto the die's actual wiring stencil. This is
  where a refusal becomes a picture rather than a sentence.
- **Provenance ledger** — every constraint the verdict rests on, sorted by
  whether it is sourced or assumed.

---

## 12. Instruments

Roughly two-thirds of this exists as working code in `audit/` with no window.
Where that is so, the script is named.

**Authoring** — program text editor · topology editor · lattice builder (grid,
King's graph, RBM, custom stencil) · coupling and bias inspector

**Sampler control** — schedule editor (warmup, samples, steps per sample) ·
chromatic block schedule, showing which spins fire together and why that is
legal · clamp and port editor · temperature and annealing · parallel chains and
seeds

**Diagnostics** — energy trace · magnetization · autocorrelation time, how many
sweeps before a sample is independent (`mixing.py`) · effective sample size
(`ess`) · R-hat (`preflight.diagnostics`) · Binder cumulant · energy histogram
and density of states · first-passage time

**Structure** — spin field · correlation function and correlation length
(`correlation_length.py`) · **structure factor S(k)** · cluster and domain
analysis · projected state space (`lib/pca.ts`) · free energy landscape
(`free_energy_surface.py`, `free_energy_sublattice.py`)

**Hardware** — placement cost curve (`placement_curve.py`) · connectivity cost
(`connectivity_cost.py`) · degree-16 band (`degree16_band.py`) · quantisation to
silicon coefficients (`quantised_for_silicon.py`)

**Verification** — exact-enumeration oracle (`audit/oracles/exact.py`,
`r15_oracle_comparison_check.py`) · **ablation runner** · diagnostic control,
which asks whether the diagnostics can report failure at all
(`diagnostic_control.py`) · receipt replay and diff

**Output** — game viewport · world and terrain · sprite and material forge ·
sequence · HP lattice protein · large-dataset table with export

### The three that carry the product

**Structure factor.** A live Fourier transform of the spin lattice. Diffuse
static when hot; a bright ring blooming out of the noise near the critical
point; sharp points below it. A standard, real measurement that happens to look
like a sensor display.

**The critical-point sweep.** Drag temperature, watch specific heat and
susceptibility spike. Onsager's exact critical point is already in the compiler,
so his answer is drawn as a reference line and the simulation's peak lands on
it, live. `critical_sweep.py` already does the measurement.

**The ablation runner.** A button that attempts to prove the user's own model is
doing nothing: zero a term, resample, compare the change against the noise floor
from a different seed. This is what separates the suite from a dashboard, and it
belongs on the front page.

---

## 13. Pre-registration

A dozen scripts in `audit/` open with "PRE-REGISTERED. Committed before the run.
The bands are NOT adjusted afterwards." That practice becomes a feature.

Before sampling, a user may state a prediction and a pass/fail criterion and
lock them. The tool records both alongside the result, and the result screen
shows the prediction as it was written, not as it might be remembered.

No commercial instrument software does this. It is the credential that makes the
suite a scientific instrument rather than a visualiser.

---

## 14. Visual system

Ice-blue on true black, hairline strokes, monospace metrics, tick-mark scales,
radial gauges, bracket frame chrome. A heads-up display.

The reason this is the correct language rather than a borrowed one: a HUD exists
to show **margin to a limit**. That is exactly what the compiler outputs —
degree against a cap, coupling against a cap, spins against the die budget,
placement effort against a ceiling. Bounded quantities are what radial gauges
are for and what bar charts are bad at.

**The rule: every element on screen encodes something real.** Tick rulers carry
real scales with units. Radials show real bounded quantities. Warning chevrons
appear only when there is a warning. Frame chrome and bracket corners may be
decoration, because they do not claim to mean anything — that is the line.

**The signature encoding: solid stroke means sourced from a published Extropic
figure, dashed stroke means this project's own assumption.** One glance at any
gauge tells you which. Nobody else can draw this, because nobody else tracks it.

Colour is spent on meaning only: ice-blue for live values, amber for warn, red
for fail. Never on decoration.

---

## 15. Honesty constraints

These are load-bearing. The project's value is that its pictures do not
overclaim, and a visual suite is where that is easiest to lose.

- **A draw is not a distribution.** The viewport states whether it is showing
  one sample, a mean over N, or a variance. These look similar and mean entirely
  different things.
- **A projection says it is one.** Showing the die as a plane with energy as
  height is honest. Showing an eleven-thousand-dimensional state space on three
  axes is also honest, but only when labelled as a projection with how much it
  captures. Conflating the two would be the one unforced error available here.
- **Provenance is never hidden.** Sourced and assumed are visually distinct
  everywhere, at all times.
- **A missing measurement reads "unavailable" with its reason.** Never a blank,
  a zero, a dash, or a guess. This already holds across the codebase.
- **No hardware claims.** Everything is simulation until someone else runs it on
  silicon.

---

## 16. Packaging

A single Windows executable: PyInstaller bundling Python, JAX, THRML and the
compiler, with a pywebview window over the WebView2 runtime already present on
Windows 11. Localhost stays as internal plumbing the user never sees or
configures. Native window, no browser chrome, nothing to start.

WebGL2 is sufficient for the lattice viewport and the game. No WebGPU
requirement, which keeps compatibility wide.

**Honest tradeoff:** JAX and SciPy are heavy. A fully self-contained build lands
in the hundreds of megabytes. The current `GibbsObservatory.exe` is 12 MB
because it is a launcher, not a bundle. That size is the real cost of "needs
nothing", and it is worth paying.

---

## 17. Build sequence

The ordering rule comes from the success criterion: the thing must be
downloadable early and stay downloadable. Packaging last is how a project like
this dies — you discover at the end that the bundler cannot carry JAX, and
nothing has ever shipped.

**Slice 0 — the pipeline, before the product.** A near-empty app: a pywebview
window over a built frontend, PyInstaller bundling Python, JAX, THRML and the
compiler, one screen that imports the compiler and prints its version. It does
almost nothing, and it is a downloadable executable. Every risk that would
otherwise surface at the end surfaces here, when it is cheap.

**Slice 1 — the frame.** Menu bar, workspace tabs, split panes, editor-type
dropdown. Existing Observatory views re-parented as editor types.

**Slice 2 — the loop.** Load from editor, paste, or file. Format sniffing. The
state machine. Two-tier compile. The REFUSED screen, designed properly.

**Slice 3 — the vertical slice: the game.** Program contract, decoder contract,
pointer-locked viewport, input as clamping. This slice proves the architecture,
because the game exercises every layer at once: compile, sample, decode, render,
play. If the pane system can hold a pointer-locked game at full speed, it can
hold anything.

**Slice 4 — the pass stack.** The modifier stack, the placement view, the
provenance ledger.

**Slice 5 — instruments.** Existing audit scripts get windows, in this order:
structure factor, critical-point sweep, ablation runner, then the rest.

**Slice 6 — decoders two and three.** Sequence, then the sprite forge.

**Slice 7 — absorb LATTICE.** Its panels become editor types, its world
generator becomes a decoder. Retire `lattice_app.py`.

From Slice 0 onward every slice ends in a rebuilt executable. There is never a
period where the answer to "can I download it" is no.

**Later, not scheduled:** HP lattice protein. Pre-registration. Arbitrary pane
docking, if it ever binds.

---

## 18. Open questions

- **The name.** "Workbench" is a placeholder. LATTICE and Gibbs Observatory both
  have equity; neither describes this.
- **Distribution.** A hundreds-of-megabytes executable needs somewhere to live.
  GitHub Releases is the obvious answer and is free; it wants deciding before
  Slice 0 ships rather than after.
- **Code signing.** An unsigned executable downloaded from the internet gets a
  SmartScreen warning on Windows. Not a blocker, but it is the first thing a
  stranger sees, and a certificate costs real money. Decide whether to pay or to
  document the warning honestly.
- **Sampler in the browser.** A WebGL compute sampler would drive the viewport at
  full frame rate, with THRML as the authoritative oracle behind it. This is how
  the game already works and the two have been checked against each other. It is
  an optimisation, not a v1 decision, and it carries a real risk: two samplers
  that can disagree.
- **Licensing of anything consulted.** Open Source Physics is GPL and this
  project is Apache 2.0. Algorithms from published papers are free to
  reimplement; OSP's source is not free to copy. Individual ComPADRE items carry
  their own terms and must be checked per item before anything is lifted.
- **Does the FastAPI backend survive packaging unchanged**, or does the bundle
  want an in-process call path instead of localhost HTTP?
