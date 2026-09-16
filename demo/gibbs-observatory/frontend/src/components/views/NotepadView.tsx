import { useCallback, useEffect, useState } from 'react'
import type { ReceiptInspect } from '../../types'

interface GateOut {
  name?: string
  gate?: string
  value?: number | string | null
  measured?: number | string | null
  limit?: number | string | null
  status?: string
  passed?: boolean
  note?: string
  assumed?: boolean
  downgraded?: boolean
}

interface ProgramResult {
  ok?: boolean
  kind?: string
  verdict?: string
  n_spins?: number
  n_nodes?: number | null
  n_couplings?: number
  mediators?: number
  mediator_count?: number | null
  bipartite?: boolean | null
  embedding?: string
  gates?: GateOut[]
  remediations?: string[]
  errors?: string[]
  candidates?: { encoding: string; state: string; reason: string }[]
  encoding_note?: string | null
  receipt_id?: string | null
  receipt_path?: string | null
  elapsed_seconds?: number
  notes?: string[]
  limits?: Record<string, string>
  place_error?: string | null
  message?: string
  label?: string
}

interface Props {
  receipt: ReceiptInspect | null
  onApplyReceipt: (id: string) => Promise<void>
}

async function postJson(url: string, body: unknown): Promise<{ status: number; data: any }> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data = await res.json().catch(() => ({}))
  return { status: res.status, data }
}

function detailMessage(data: any): string {
  if (!data) return 'request failed'
  if (typeof data.detail === 'string') return data.detail
  if (data.detail?.message) return data.detail.message
  if (data.message) return data.message
  return JSON.stringify(data.detail ?? data)
}

export function NotepadView({ receipt, onApplyReceipt }: Props) {
  const [yaml, setYaml] = useState('')
  const [seed, setSeed] = useState('')
  const [busy, setBusy] = useState<string | null>(null)
  const [result, setResult] = useState<ProgramResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [compiledId, setCompiledId] = useState<string | null>(null)
  const [tsuOk, setTsuOk] = useState<boolean | null>(null)

  const seedFromReceipt = useCallback(async (id: string | null) => {
    if (!id) {
      const res = await fetch('/api/program/status')
      const st = await res.json().catch(() => ({}))
      setTsuOk(!!st.importable)
      // default seed via empty receipt endpoint fallback
      const spec = await fetch('/api/receipts/small/spec').then((r) => r.json())
      const text = spec.spec_yaml || spec.default_seed || ''
      setYaml(text)
      setSeed(text)
      return
    }
    const [spec, st] = await Promise.all([
      fetch(`/api/receipts/${id}/spec`).then((r) => r.json()),
      fetch('/api/program/status').then((r) => r.json()),
    ])
    setTsuOk(!!st.importable)
    const text =
      (receipt?.spec_yaml as string | undefined) ||
      spec.spec_yaml ||
      spec.default_seed ||
      ''
    setYaml(text)
    setSeed(text)
  }, [receipt?.spec_yaml])

  useEffect(() => {
    void seedFromReceipt(receipt?.id ?? null)
  }, [receipt?.id, seedFromReceipt])

  const runPreflight = async () => {
    setBusy('preflight')
    setError(null)
    setResult(null)
    try {
      const { status, data } = await postJson('/api/program/preflight', {
        yaml,
        allow_assumed: false,
      })
      if (status >= 400) {
        setError(detailMessage(data))
        setResult(null)
      } else {
        setResult(data as ProgramResult)
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(null)
    }
  }

  const runCompile = async () => {
    setBusy('compile')
    setError(null)
    setResult(null)
    setCompiledId(null)
    try {
      const { status, data } = await postJson('/api/program/compile', {
        yaml,
        allow_assumed: false,
        target: 'z1',
        receipt_id: 'notepad',
      })
      if (status >= 400) {
        setError(detailMessage(data))
        setResult(null)
        return
      }
      setResult(data as ProgramResult)
      if (data.ok && data.receipt_id) {
        setCompiledId(data.receipt_id)
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(null)
    }
  }

  const runApply = async () => {
    if (!compiledId) {
      setError('Apply refused, compile a COMPILED receipt first (no silent apply).')
      return
    }
    setBusy('apply')
    setError(null)
    try {
      const { status, data } = await postJson('/api/program/apply', {
        receipt_id: compiledId,
      })
      if (status >= 400) {
        setError(detailMessage(data))
        return
      }
      setResult(data as ProgramResult)
      await onApplyReceipt(compiledId)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(null)
    }
  }

  const revert = () => {
    setYaml(seed)
    setError(null)
    setResult(null)
    setCompiledId(null)
  }

  const gates = result?.gates ?? []
  const mediators = result?.mediator_count ?? result?.mediators
  const bipartite = result?.bipartite

  return (
    <div className="view notepad-view">
      <div className="notepad-head">
        <div>
          <h3>Thermodynamic Program</h3>
          <p className="footnote">
            Gill / Extropic usage: a stochastic program aimed at a sampler, {' '}
            <strong>not</strong> “thermodynamic programming” as a research slogan.
          </p>
        </div>
        <div className="notepad-meta mono">
          tsu {tsuOk == null ? '…' : tsuOk ? 'ready' : 'missing'}
          {receipt?.id ? ` · seeded from ${receipt.id}` : ''}
        </div>
      </div>

      <div className="notepad-actions">
        <button type="button" className="btn" disabled={!!busy} onClick={() => void runPreflight()}>
          {busy === 'preflight' ? 'Preflight…' : 'Preflight'}
        </button>
        <button type="button" className="btn primary" disabled={!!busy} onClick={() => void runCompile()}>
          {busy === 'compile' ? 'Compile…' : 'Compile (ideal-first)'}
        </button>
        <button
          type="button"
          className="btn accent"
          disabled={!!busy || !compiledId}
          onClick={() => void runApply()}
          title={compiledId ? `Apply receipt ${compiledId}` : 'Requires successful compile receipt'}
        >
          {busy === 'apply' ? 'Apply…' : 'Apply'}
        </button>
        <button type="button" className="btn" disabled={!!busy} onClick={revert}>
          Revert
        </button>
      </div>

      <div className="notepad-grid">
        <textarea
          className="notepad-editor mono"
          spellCheck={false}
          value={yaml}
          onChange={(e) => {
            setYaml(e.target.value)
            setCompiledId(null)
          }}
          aria-label="Thermodynamic Program YAML"
        />
        <div className="notepad-out">
          <div className="view-caption">Output · gate table · mediators · bipartite · errors</div>
          {error ? <div className="status-error notepad-err">{error}</div> : null}
          {result ? (
            <>
              <div className="mission-strip notepad-strip">
                <div>
                  <span className="k">verdict</span>
                  <span className={`v mono ${result.ok || result.verdict === 'ok' || result.verdict === 'COMPILED' ? 'ok' : 'bad'}`}>
                    {result.verdict ?? ', '}
                  </span>
                </div>
                <div>
                  <span className="k">mediators</span>
                  <span className="v mono">{mediators ?? ', '}</span>
                </div>
                <div>
                  <span className="k">bipartite</span>
                  <span className="v mono">
                    {bipartite == null ? ', ' : bipartite ? 'yes' : 'no'}
                  </span>
                </div>
                <div>
                  <span className="k">spins</span>
                  <span className="v mono">{result.n_spins ?? result.n_nodes ?? ', '}</span>
                </div>
                <div>
                  <span className="k">elapsed</span>
                  <span className="v mono">
                    {result.elapsed_seconds != null ? `${result.elapsed_seconds}s` : ', '}
                  </span>
                </div>
              </div>
              {result.receipt_id ? (
                <p className="dim mono">receipt: {result.receipt_id}</p>
              ) : null}
              {gates.length > 0 ? (
                <table className="gate-table compact">
                  <thead>
                    <tr>
                      <th>gate</th>
                      <th>status</th>
                      <th>value</th>
                      <th>limit</th>
                    </tr>
                  </thead>
                  <tbody>
                    {gates.map((g, i) => {
                      const name = g.name ?? g.gate ?? `g${i}`
                      const status =
                        g.status ??
                        (g.passed == null ? ', ' : g.passed ? 'ok' : 'fail')
                      const val = g.value ?? g.measured ?? ', '
                      return (
                        <tr key={name} className={String(status)}>
                          <td className="mono">{name}</td>
                          <td>{status}</td>
                          <td className="mono">{String(val)}</td>
                          <td className="mono">{String(g.limit ?? ', ')}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              ) : (
                <p className="empty-hint">No gate rows in this response.</p>
              )}
              {(result.candidates ?? []).length > 0 ? (
                <div className="cand-table" style={{ marginTop: 8 }}>
                  <div className="view-caption">Encoding candidates</div>
                  <table>
                    <thead>
                      <tr>
                        <th>encoding</th>
                        <th>state</th>
                        <th>note</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.candidates!.map((c) => (
                        <tr key={`${c.encoding}-${c.state}`}>
                          <td>
                            <code>{c.encoding}</code>
                          </td>
                          <td>
                            <span
                              className={
                                c.state === 'SELECTED'
                                  ? 'pill ok'
                                  : c.state === 'VIABLE_NOT_SELECTED'
                                    ? 'pill'
                                    : 'pill bad'
                              }
                            >
                              {c.state}
                            </span>
                          </td>
                          <td className="muted">{c.reason || ', '}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {result.encoding_note ? (
                    <p className="empty-hint" style={{ marginTop: 6 }}>
                      {result.encoding_note}
                    </p>
                  ) : null}
                </div>
              ) : null}
              {(result.errors ?? []).length > 0 ? (

                <ul className="err-list">
                  {result.errors!.map((e) => (
                    <li key={e}>{e}</li>
                  ))}
                </ul>
              ) : null}
              {(result.remediations ?? []).length > 0 ? (
                <ul className="dim">
                  {result.remediations!.map((e) => (
                    <li key={e}>{e}</li>
                  ))}
                </ul>
              ) : null}
              {(result.notes ?? []).map((n) => (
                <p key={n} className="empty-hint">
                  {n}
                </p>
              ))}
              {result.label ? <p className="claim-chip">{result.label}</p> : null}
            </>
          ) : (
            <p className="empty-hint">
              Preflight reports gates without writing a receipt. Compile (ideal-first) writes{' '}
              <span className="mono">receipts/notepad</span> when COMPILED. Apply loads that
              receipt into the sampler, never silently.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
