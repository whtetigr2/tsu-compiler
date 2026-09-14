# Help + walkthrough — shipping notes

**Version:** 0.5.1  
**Date:** 2026-09-13

## What shipped

### In-app guided walkthrough (`frontend/src/components/Walkthrough.tsx`)

- Spotlight / coach-mark overlay (dark engineering style) with ~13 steps covering
  mission, program bar, receipts, Overview→Spins→Couplings→Connectivity→Schedule,
  footer transport, Sci|Prog, notepad, claim hygiene, snapshot.
- Optional `onNavigate(view)` switches the left rail per step.
- **First visit:** auto-starts once if `localStorage` key
  `gibbs-observatory-walkthrough-v1` is unset; Finish / Skip / Esc sets the key.
- **Restart:** ☰ → Help → Start walkthrough…, or Help modal → Walkthrough tab.
- Controls: Skip / Back / Next / Finish; Esc closes; ← → Enter also work.
- Targets use `data-tour` on brand, menu, program bar, nav items, lang toggle,
  honesty badge, footer.

### Help modal (`HelpModal.tsx`)

Tabs: **About** | **Walkthrough** (launch button) | **UI map** | **Receipts & sampling** | **Claim hygiene**  
(`ClaimHygienePanel` unchanged.)

### TopMenu Help

- Start walkthrough…
- User guide… (opens Help on About)
- Claim hygiene…
- Keyboard shortcuts (small Esc / arrows / ☰ list)

### Docs (clone-and-use)

- `WALKTHROUGH.md` — markdown mirror of the tour
- `README.md` — repo-relative setup (`PYTHONPATH=.`), optional sibling `tsu` /
  `TSU_ROOT`, brief Windows activate notes; links to walkthrough + in-app Help
- Softened user-facing `/workspace/...` paths in README / STATUS / program_service
  error text; `TSU_ROOT` + sibling discovery for notepad compile

## Verification

```bash
cd frontend && npm run build
PYTHONPATH=. pytest backend/tests -q
# with tsu: PYTHONPATH=.:../tsu-compiler pytest backend/tests -q
```

## Honesty reminders (unchanged)

- No silicon / joule claims
- Mediated β FIXED language
- Existing sampling + help-open paths preserved (Help still opens; hygiene tab still works)
