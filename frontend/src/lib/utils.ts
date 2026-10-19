import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Readable numbers: fixed decimals for small values, never scientific notation. */
export function num(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  const a = Math.abs(v)
  if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (a >= 1) return Number(v.toPrecision(Math.max(digits + 1, 3))).toString()  // e.g. 24.81, 2.117
  if (a === 0) return '0'
  // small magnitudes: keep `digits` decimals, or enough to show two significant figures
  const decimals = a >= 10 ** -digits ? digits : Math.min(8, Math.ceil(-Math.log10(a)) + 1)
  return v.toFixed(decimals)
}

export function withUnit(v: number | null | undefined, unit: string, digits = 3): string {
  const s = num(v, digits)
  return unit && s !== '—' ? `${s} ${unit}` : s
}

export function signed(v: number | null | undefined, unit = '', digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  const s = num(Math.abs(v), digits)
  return `${v > 0 ? '+' : v < 0 ? '−' : '±'}${s}${unit ? ` ${unit}` : ''}`
}
