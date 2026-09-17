import type { GateRow } from '../types'

export interface CompileResult {
  verdict?: string
  ok?: boolean
  placed?: boolean
  /** The compiler's own answer to "did any gate actually refuse anything?".
   *  Absent on the preflight payload, which predates it. */
  hardware_evaluated?: boolean
  /** The compiler's own sentence about why the candidates ended where they
   *  did. Shown verbatim rather than reconstructed from a verdict string. */
  repset_reason?: string
  place_error?: string | null
  placement_escalated?: boolean
  placement_attempts?: { restarts: number; iters: number; placed: boolean; seconds: number }[]
  gates?: GateRow[]
  remediations?: string[]
  n_spins?: number
  n_couplings?: number
  max_degree?: number
  mediators?: number
  elapsed_seconds?: number
}

/**
 * What the compiler said, and which kind of "no" it is.
 *
 * Anyone can print "compile failed". The distinction this panel exists to draw
 * is that there are THREE different refusals and they mean different things:
 *
 *   A GATE REFUSED        the model does not fit the hardware. Real limit.
 *   THE SEARCH RAN OUT    no embedding was found in the budget allowed. The
 *                         model may fit perfectly well. Not a hardware limit,
 *                         and saying so would be a claim about Extropic's
 *                         silicon made on the basis of a timer (R27).
 *   THE COMPILER FAULTED  our bug. Says nothing about the model at all.
 *
 * Conflating the first two is the representation-versus-hardware confusion this
 * compiler exists to prevent, and it has been found in the codebase twice.
 *
 * Every cap also carries its provenance. A limit sourced from Extropic's
 * published figures is drawn solid; one this project assumed is drawn dashed
 * and labelled. A refusal resting on an assumption is a weaker claim than one
 * resting on a published number, and the reader is entitled to see which.
 */
export function RefusalPanel({ result }: { result: CompileResult | null }) {
  if (!result) return null

  const gates = result.gates ?? []
  const failed = gates.filter((g) => g.status === 'fail' || g.passed === false)

  if (result.verdict === 'ok' || result.verdict === 'COMPILED') {
    return <Compiled result={result} />
  }

  // Which refusal this is turns on whether the hardware question was ever
  // REACHED, not on whether an embedding was found. Branching on `placed`
  // alone reads a gate refusal and a search that ran out of budget as the
  // same event, because both leave a model unplaced.
  //
  // The compile path answers this outright: the compiler decides it and says
  // so in `hardware_evaluated`, and names the case `EFFORT`. The preflight
  // path predates both fields, and there the answer is inferred the way it
  // always was, from an unplaced model with every gate passing. Inferring it
  // when the explicit answer is available is what let a HARDWARE refusal
  // render as "not a hardware limit".
  const effort = result.hardware_evaluated === undefined
    ? result.placed === false && failed.length === 0
    : result.verdict === 'EFFORT' || result.hardware_evaluated === false

  if (effort) return <EffortRefusal result={result} />
  if (failed.length) return <GateRefusal result={result} failed={failed} gates={gates} />
  return <UnknownRefusal result={result} />
}

function Compiled({ result }: { result: CompileResult }) {
  const attempts = result.placement_attempts ?? []
  return (
    <div className="refusal compiled">
      <div className="refusal-head">
        <span className="refusal-kind ok">COMPILED</span>
        <span className="refusal-lead">
          It fits. Every gate passed and an embedding was found.
        </span>
      </div>
      <dl className="refusal-facts">
        <Fact label="spins" value={result.n_spins} />
        <Fact label="couplings" value={result.n_couplings} />
        <Fact label="max degree" value={result.max_degree} />
        <Fact label="mediators" value={result.mediators} />
      </dl>
      {result.placement_escalated && attempts.length > 1 ? (
        <p className="refusal-note">
          Placement needed more than the cheapest budget. It was refused at{' '}
          {attempts[0].restarts} restarts and {attempts[0].iters.toLocaleString()}{' '}
          iterations, then placed at {attempts[attempts.length - 1].restarts} and{' '}
          {attempts[attempts.length - 1].iters.toLocaleString()}. That first
          refusal was about search effort, not about the hardware, which is why
          it was retried rather than reported.
        </p>
      ) : null}
    </div>
  )
}

function GateRefusal({ result, failed, gates }: {
  result: CompileResult
  failed: GateRow[]
  gates: GateRow[]
}) {
  return (
    <div className="refusal gate">
      <div className="refusal-head">
        <span className="refusal-kind bad">DOES NOT FIT</span>
        <span className="refusal-lead">
          {failed.length === 1
            ? 'One hardware limit is exceeded. This is a real limit, not a search that gave up.'
            : `${failed.length} hardware limits are exceeded. These are real limits, not a search that gave up.`}
        </span>
      </div>

      {failed.map((g) => <GateBar key={String(g.gate ?? g.name)} gate={g} />)}

      {result.repset_reason ? (
        <p className="refusal-note">{result.repset_reason}</p>
      ) : null}

      {result.remediations?.length ? (
        <>
          <h5>What would change it</h5>
          <ul className="refusal-fixes">
            {result.remediations.map((r) => <li key={r}>{r}</li>)}
          </ul>
        </>
      ) : null}

      <details className="refusal-all">
        <summary>All {gates.length} gates</summary>
        {gates.map((g) => <GateBar key={`all-${String(g.gate ?? g.name)}`} gate={g} />)}
      </details>
    </div>
  )
}

function EffortRefusal({ result }: { result: CompileResult }) {
  const attempts = result.placement_attempts ?? []
  return (
    <div className="refusal effort">
      <div className="refusal-head">
        <span className="refusal-kind warn">NOT PLACED</span>
        <span className="refusal-lead">
          The layout search ran out of budget. <strong>This is not a hardware
          limit.</strong> Every gate passed, so nothing here says the model does
          not fit.
        </span>
      </div>
      {attempts.length ? (
        <table className="refusal-attempts">
          <thead>
            <tr><th>restarts</th><th>iterations</th><th>seconds</th><th /></tr>
          </thead>
          <tbody>
            {attempts.map((a, i) => (
              <tr key={i}>
                <td>{a.restarts}</td>
                <td>{a.iters.toLocaleString()}</td>
                <td>{a.seconds.toFixed(1)}</td>
                <td>{a.placed ? 'placed' : 'not found'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      {result.repset_reason ? (
        <p className="refusal-note">{result.repset_reason}</p>
      ) : null}
      {result.place_error ? (
        <p className="refusal-note mono">{result.place_error}</p>
      ) : null}
      <p className="refusal-note">
        More restarts or more iterations may find an embedding. A different
        encoding may make one easier to find. Neither is a statement about
        Extropic&rsquo;s silicon.
      </p>
    </div>
  )
}

function UnknownRefusal({ result }: { result: CompileResult }) {
  return (
    <div className="refusal unknown">
      <div className="refusal-head">
        <span className="refusal-kind warn">
          {String(result.verdict ?? 'unavailable').toUpperCase()}
        </span>
        <span className="refusal-lead">
          The compiler returned a verdict this panel does not recognise, so it is
          shown as given rather than guessed at.
        </span>
      </div>
      {result.place_error ? (
        <p className="refusal-note mono">{result.place_error}</p>
      ) : null}
    </div>
  )
}

/** One gate, with its measured value against its cap, and where the cap is from. */
function GateBar({ gate }: { gate: GateRow }) {
  const name = String(gate.gate ?? gate.name ?? 'gate')
  const measured = typeof gate.measured === 'number'
    ? gate.measured
    : typeof gate.value === 'number' ? gate.value : null
  const limit = typeof gate.limit === 'number' ? gate.limit : null
  const failed = gate.status === 'fail' || gate.passed === false
  const assumed = Boolean(gate.assumed)

  const over = measured != null && limit != null && limit > 0
    ? measured / limit
    : null
  const pct = over != null ? Math.min(100, (over / Math.max(over, 1)) * 100) : 0
  const capPct = over != null ? Math.min(100, (1 / Math.max(over, 1)) * 100) : 100

  return (
    <div className={`gate-row ${failed ? 'failed' : 'passed'}`}>
      <div className="gate-row-head">
        <span className="gate-name">{name}</span>
        {assumed ? (
          <span className="chip assumed" title="This project's working value, not a published Extropic figure">
            ASSUMED
          </span>
        ) : (
          <span className="chip sourced" title="Sourced from Extropic's published figures">
            SOURCED
          </span>
        )}
        <span className="gate-numbers">
          {measured != null ? measured.toLocaleString() : 'unavailable'}
          {' / '}
          <span className="gate-cap">{limit != null ? limit.toLocaleString() : 'unavailable'}</span>
          {over != null && over > 1 ? (
            <span className="gate-over"> {((over - 1) * 100).toFixed(0)}% over</span>
          ) : null}
        </span>
      </div>
      <div className={`gate-track ${assumed ? 'assumed' : ''}`}>
        <div className={`gate-fill ${failed ? 'bad' : 'ok'}`} style={{ width: `${pct}%` }} />
        <div className="gate-cap-mark" style={{ left: `${capPct}%` }} />
      </div>
      {gate.note ? <p className="gate-note">{String(gate.note)}</p> : null}
    </div>
  )
}

function Fact({ label, value }: { label: string; value: number | undefined }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value != null ? value.toLocaleString() : 'unavailable'}</dd>
    </div>
  )
}
