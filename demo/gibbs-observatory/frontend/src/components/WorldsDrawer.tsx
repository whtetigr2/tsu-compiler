interface Props {
  open: boolean
  onClose: () => void
}

export function WorldsDrawer({ open, onClose }: Props) {
  if (!open) return null
  return (
    <div className="drawer-backdrop" onClick={onClose} role="presentation">
      <div
        className="drawer"
        role="dialog"
        aria-label="World Studio"
        onClick={(e) => e.stopPropagation()}
      >
        <header>
          <h3>World Studio</h3>
          <button type="button" className="btn" onClick={onClose}>
            Close
          </button>
        </header>
        <p>
          <strong>World Studio demoted, Phase 2+</strong>
        </p>
        <p className="empty-hint">
          Worlds / world-gen are intentionally off the hero path. Gibbs Observatory
          inspects compiled receipts + live THRML sampling. World generation will
          return as a drawer tool, not the product identity.
        </p>
      </div>
    </div>
  )
}
