import * as React from 'react'

/** Follows the system appearance so charts can use the palette steps chosen for each mode. */
export function useDark(): boolean {
  const query = '(prefers-color-scheme: dark)'
  const [dark, setDark] = React.useState(() => window.matchMedia?.(query).matches ?? false)
  React.useEffect(() => {
    const m = window.matchMedia?.(query)
    if (!m) return
    const on = (e: MediaQueryListEvent) => setDark(e.matches)
    m.addEventListener('change', on)
    return () => m.removeEventListener('change', on)
  }, [])
  return dark
}

/** Validated categorical palette (blue, orange, aqua first) with separate light/dark steps. */
export function usePalette() {
  const dark = useDark()
  return dark
    ? { series: ['#3987e5', '#d95926', '#199e70', '#c98500'], muted: '#7c7c80', grid: 'rgba(255,255,255,0.09)', ink2: '#a1a1a6', surface: '#1c1c1e' }
    : { series: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'], muted: '#8a8985', grid: 'rgba(0,0,0,0.08)', ink2: '#6b6b70', surface: '#ffffff' }
}
