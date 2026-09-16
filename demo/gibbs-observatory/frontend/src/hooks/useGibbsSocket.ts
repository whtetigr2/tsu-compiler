import { useCallback, useEffect, useRef, useState } from 'react'
import type { BatchPayload, GraphPayload, SimParams } from '../types'

export type SocketStatus =
  | 'connecting'
  | 'open'
  | 'reconnecting'
  | 'closed'

const MAX_BACKOFF_MS = 8000
const BASE_BACKOFF_MS = 400

function wsUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${window.location.host}/ws/stream`
}

export function useGibbsSocket() {
  const wsRef = useRef<WebSocket | null>(null)
  const intentionalClose = useRef(false)
  const attemptRef = useRef(0)
  const reconnectTimer = useRef<number | null>(null)
  const [status, setStatus] = useState<SocketStatus>('connecting')
  const [running, setRunning] = useState(false)
  const [graph, setGraph] = useState<GraphPayload | null>(null)
  const [batch, setBatch] = useState<BatchPayload | null>(null)
  const [error, setError] = useState<string | null>(null)

  const clearReconnectTimer = () => {
    if (reconnectTimer.current != null) {
      window.clearTimeout(reconnectTimer.current)
      reconnectTimer.current = null
    }
  }

  const connect = useCallback(() => {
    clearReconnectTimer()
    // Close any dangling socket without scheduling another reconnect
    if (wsRef.current) {
      try {
        wsRef.current.onclose = null
        wsRef.current.onerror = null
        wsRef.current.onmessage = null
        wsRef.current.onopen = null
        wsRef.current.close()
      } catch {
        /* ignore */
      }
      wsRef.current = null
    }

    const ws = new WebSocket(wsUrl())
    wsRef.current = ws
    setStatus(attemptRef.current === 0 ? 'connecting' : 'reconnecting')

    ws.onopen = () => {
      attemptRef.current = 0
      setStatus('open')
      // Clear transient connect noise; keep sampler errors until pause/reset clears them
      setError((prev) =>
        prev && (prev.startsWith('WebSocket') || prev.startsWith('Disconnected'))
          ? null
          : prev,
      )
    }

    ws.onclose = (ev) => {
      wsRef.current = null
      setRunning(false)
      if (intentionalClose.current) {
        setStatus('closed')
        return
      }
      // Browsers often fire a generic error event right before close during
      // proxy/HMR blips, don't leave a sticky "WebSocket error".
      const reason =
        ev.code === 1000
          ? null
          : `Disconnected (code ${ev.code}${ev.reason ? `: ${ev.reason}` : ''}), reconnecting…`
      if (reason) setError(reason)
      else setError(null)
      setStatus('reconnecting')
      const attempt = attemptRef.current++
      const delay = Math.min(MAX_BACKOFF_MS, BASE_BACKOFF_MS * 2 ** attempt)
      reconnectTimer.current = window.setTimeout(() => connect(), delay)
    }

    ws.onerror = () => {
      // onclose always follows; avoid sticky vague "WebSocket error"
      setError((prev) => prev ?? 'WebSocket transport issue, waiting for reconnect…')
    }

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data)
        if (msg.type === 'graph') {
          setGraph(msg as GraphPayload)
          setError(null)
        } else if (msg.type === 'batch') {
          setBatch(msg as BatchPayload)
        } else if (msg.type === 'status') {
          setRunning(!!msg.running)
          // Successful control message → clear disconnect sticky text
          if (!msg.running) {
            setError((prev) =>
              prev &&
              (prev.startsWith('WebSocket') ||
                prev.startsWith('Disconnected') ||
                prev.includes('reconnect'))
                ? null
                : prev,
            )
          }
        } else if (msg.type === 'error') {
          setError(String(msg.message ?? 'sampler error'))
          setRunning(false)
        } else if (msg.type === 'pong') {
          /* keepalive */
        }
      } catch {
        /* ignore malformed */
      }
    }
  }, [])

  useEffect(() => {
    intentionalClose.current = false
    connect()
    return () => {
      intentionalClose.current = true
      clearReconnectTimer()
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connect])

  const send = useCallback((payload: Record<string, unknown>) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload))
      return true
    }
    setError('Not connected, message queued only after reconnect + reset')
    return false
  }, [])

  const reset = useCallback(
    (params: SimParams) => {
      setBatch(null)
      setError(null)
      const body: Record<string, unknown> = { type: 'reset', ...params }
      if (params.receipt_id) {
        body.receipt_id = params.receipt_id
      } else {
        body.receipt_id = null
      }
      send(body)
    },
    [send],
  )

  const updateParams = useCallback(
    (patch: Partial<SimParams>) => send({ type: 'params', ...patch }),
    [send],
  )

  const run = useCallback(() => {
    setError(null)
    setRunning(true)
    send({ type: 'run' })
  }, [send])

  const pause = useCallback(() => {
    setRunning(false)
    // Clear sticky transport/sampler banner on intentional pause
    setError(null)
    send({ type: 'pause' })
  }, [send])

  const step = useCallback(() => {
    setError(null)
    send({ type: 'step', n: 1 })
  }, [send])

  const clearError = useCallback(() => setError(null), [])

  return {
    status,
    running,
    graph,
    batch,
    error,
    clearError,
    reset,
    updateParams,
    run,
    pause,
    step,
  }
}
