interface Props {
  title: string
  phase: string
  blurb: string
}

export function StubPanel({ title, phase, blurb }: Props) {
  return (
    <div className="stub-panel">
      <h3>{title}</h3>
      <span className="chip">{phase}</span>
      <p>{blurb}</p>
      <p className="empty-hint">
        Nav entry reserved. Coming in {phase}. Not on the hero path for Phase 1.
      </p>
    </div>
  )
}
