interface Props {
  value: number | null | undefined
  reason?: string | null
  unit?: string
  digits?: number
}

/**
 * A number, or why there is no number.
 *
 * Never renders nothing. A blank where a statistic belongs reads as a layout
 * bug, and a reader cannot tell it apart from a value of zero. The project's
 * rule is that a missing measurement says "unavailable" and says why, so the
 * reason rides along as a tooltip instead of being dropped.
 *
 * This exists because effective sample size is now null most of the time: the
 * compiler's estimator refuses below N/tau >= 5000 and the live history is 256
 * samples. That refusal is the honest answer and it has to be legible.
 */
export function StatValue({ value, reason, unit, digits = 2 }: Props) {
  if (value == null || !Number.isFinite(value)) {
    return (
      <span className="stat-unavailable" title={reason ?? undefined}>
        unavailable
      </span>
    )
  }
  return (
    <span className="stat-value">
      {value.toFixed(digits)}
      {unit ? <span className="stat-unit">{unit}</span> : null}
    </span>
  )
}
