import type { ReceiptInspect } from '../types'

interface Props {
  receipt: ReceiptInspect | null
  fallbackBanner?: string | null
}

export function ProgramBar({ receipt, fallbackBanner }: Props) {
  const verdict = receipt?.verdict ?? 'unavailable'
  const beta = receipt?.beta
  const betaFixed = receipt?.beta_fixed
  return (
    <div className="program-bar" data-tour="program-bar">
      <div className="pb-item path" title={receipt?.path}>
        <span className="k">receipt</span>
        <span className="v mono">{receipt ? `receipts/${receipt.id}` : ', (preset)'}</span>
      </div>
      <div className="pb-item">
        <span className="k">encoding</span>
        <span className="v">{receipt?.encoding ?? 'unavailable'}</span>
      </div>
      <div className="pb-item">
        <span className="k">β</span>
        <span className="v">
          {beta != null ? beta : 'unavailable'}
          {betaFixed ? <span className="chip fixed">FIXED</span> : null}
        </span>
      </div>
      <div className="pb-item">
        <span className="k">kernel</span>
        <span className="v">{receipt?.kernel ?? 'unavailable'}</span>
      </div>
      <div className="pb-item">
        <span className={`verdict-chip ${String(verdict).toLowerCase()}`}>{verdict}</span>
      </div>
      {fallbackBanner ? <div className="pb-banner">{fallbackBanner}</div> : null}
    </div>
  )
}
