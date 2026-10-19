import { AlertTriangle, CircleAlert, FlaskConical, Info, Radio, Ruler, Globe } from 'lucide-react'
import * as React from 'react'
import { Badge } from '@/components/ui/badge'
import type { Dataset } from '@/lib/api'
import { cn } from '@/lib/utils'

export function PageHeader({ eyebrow, title, children }: { eyebrow?: string; title: string; children?: React.ReactNode }) {
  return (
    <header className="mb-10 max-w-3xl">
      {eyebrow && <p className="mb-2 text-caption font-semibold text-accent-ink">{eyebrow}</p>}
      <h1 className="text-display text-ink">{title}</h1>
      {children && <div className="mt-4 text-[1.0625rem] leading-relaxed text-ink-2">{children}</div>}
    </header>
  )
}

export function SectionTitle({ title, description, action }: { title: string; description?: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h2 className="text-title text-ink">{title}</h2>
        {description && <p className="mt-1.5 max-w-3xl text-caption text-ink-2">{description}</p>}
      </div>
      {action}
    </div>
  )
}

const TYPE_BADGE: Record<Dataset['type'], { label: string; tone: 'warning' | 'accent' | 'success' | 'neutral'; icon: React.ReactNode }> = {
  SYNTHETIC: { label: 'Synthetic demo data', tone: 'warning', icon: <FlaskConical /> },
  SIMULATION: { label: 'EM simulation data', tone: 'accent', icon: <Radio /> },
  MEASUREMENT: { label: 'Measured data', tone: 'success', icon: <Ruler /> },
  PUBLIC: { label: 'Public data', tone: 'neutral', icon: <Globe /> },
}

export function ProvenanceBadge({ type }: { type: Dataset['type'] }) {
  const b = TYPE_BADGE[type]
  return (
    <Badge tone={b.tone}>
      {b.icon}
      {b.label}
    </Badge>
  )
}

export function Notice({ tone = 'info', title, children }: { tone?: 'info' | 'warning' | 'danger'; title?: string; children: React.ReactNode }) {
  const styles = {
    info: 'bg-accent-soft text-accent-ink',
    warning: 'bg-warning-soft text-warning',
    danger: 'bg-danger-soft text-danger',
  }[tone]
  const Icon = tone === 'info' ? Info : tone === 'warning' ? AlertTriangle : CircleAlert
  return (
    <div role={tone === 'danger' ? 'alert' : 'status'} className={cn('flex gap-3 rounded-xl px-4 py-3 text-caption', styles)}>
      <Icon className="mt-0.5 size-4 shrink-0" />
      <div>
        {title && <span className="font-semibold">{title} </span>}
        {children}
      </div>
    </div>
  )
}

export function DataNotice({ dataset }: { dataset: Dataset }) {
  if (dataset.type === 'SYNTHETIC')
    return (
      <Notice tone="warning" title="Synthetic demonstration data.">
        Generated from a documented formula with noise. Results show that the workflow runs; they say nothing about real
        antenna accuracy.
      </Notice>
    )
  return (
    <Notice title="Electromagnetic simulation data.">
      Models were trained on {dataset.label} ({dataset.sample_count.toLocaleString()} HFSS-simulated designs). Predictions
      are ML estimates of the simulated response — not new simulations or measurements.
    </Notice>
  )
}

export function EmptyState({ icon, title, children, action }: { icon: React.ReactNode; title: string; children: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-start gap-3 rounded-[var(--radius-card)] border border-dashed border-hairline-strong px-6 py-8">
      <div className="flex size-10 items-center justify-center rounded-xl bg-surface-2 text-ink-2 [&_svg]:size-5">{icon}</div>
      <div>
        <p className="text-headline text-ink">{title}</p>
        <p className="mt-1 max-w-xl text-caption text-ink-2">{children}</p>
      </div>
      {action}
    </div>
  )
}

export function Stat({ label, value, sub, hint }: { label: string; value: React.ReactNode; sub?: React.ReactNode; hint?: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="flex items-center gap-1.5 text-micro uppercase text-ink-3">
        {label}
        {hint}
      </div>
      <div className="text-figure mt-1 truncate text-ink">{value}</div>
      {sub && <div className="mt-1 text-caption text-ink-2">{sub}</div>}
    </div>
  )
}
