import type { ReceiptInspect } from '../types'
import { FALLBACK_CLAIM_BADGES, STANDING_PROHIBITIONS } from '../lib/snapshot'

interface Props {
  receipt: ReceiptInspect | null
  compact?: boolean
}

export function ClaimHygienePanel({ receipt, compact = false }: Props) {
  const badges = receipt?.claim_badges?.length
    ? receipt.claim_badges
    : FALLBACK_CLAIM_BADGES
  const betaFixed = Boolean(receipt?.beta_fixed)
  const ess = receipt?.verification?.ess
  const essAvailable = Boolean(ess?.available)

  return (
    <div className={`claim-hygiene ${compact ? 'compact' : ''}`}>
      {!compact ? (
        <p className="empty-hint">
          Always honest: this panel lists standing prohibitions and live claim
          badges. Software / THRML only, no silicon, no energy claims.
        </p>
      ) : null}

      <h4>Standing prohibitions</h4>
      <ul className="prose-list claim-list">
        {STANDING_PROHIBITIONS.map((p) => (
          <li key={p.id}>{p.text}</li>
        ))}
      </ul>

      <h4>Live claim badges</h4>
      <div className="badges">
        {badges.map((b) => (
          <span key={b} className="badge">
            {b}
          </span>
        ))}
      </div>

      <div className="live-flags mono">
        <div className="flag ok">runtime: software / THRML+JAX</div>
        <div className="flag ok">silicon: none</div>
        <div className={`flag ${betaFixed ? 'ok' : 'info'}`}>
          β: {receipt?.beta ?? ', '}
          {betaFixed ? ' FIXED (mediated)' : ' editable (unmediated)'}
        </div>
        <div className={`flag ${essAvailable ? 'ok' : 'warn'}`}>
          ESS:{' '}
          {essAvailable
            ? `available (${ess?.value ?? ', '})`
            : ess?.reason
              ? `unavailable, ${ess.reason}`
              : 'diagnostic only / contract check'}
        </div>
        <div className="flag ok">energy: simulation traces only, no joule claims</div>
      </div>
    </div>
  )
}
