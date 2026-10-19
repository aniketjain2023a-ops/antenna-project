import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { Diagnostics, Experiment } from '@/lib/api'
import { usePalette } from '@/lib/use-dark'
import { num } from '@/lib/utils'

function ChartFrame({ title, caption, children, height = 280 }: { title: string; caption?: string; children: React.ReactElement; height?: number }) {
  return (
    <figure className="min-w-0">
      <figcaption className="mb-3 text-[0.9375rem] font-semibold tracking-[-0.01em] text-ink">{title}</figcaption>
      <div style={{ height }} className="w-full">
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
      {caption && <p className="mt-2 text-caption text-ink-2">{caption}</p>}
    </figure>
  )
}

function TooltipBox({ lines }: { lines: string[] }) {
  return (
    <div className="rounded-lg border border-hairline bg-surface px-3 py-2 text-caption text-ink shadow-lg">
      {lines.map((l) => (
        <div key={l} className="tabular">
          {l}
        </div>
      ))}
    </div>
  )
}

const axisLabel = (value: string, axis: 'x' | 'y') =>
  axis === 'x'
    ? { value, position: 'insideBottom' as const, offset: -4 }
    : { value, angle: -90, position: 'insideLeft' as const, style: { textAnchor: 'middle' as const } }

export function ActualVsPredicted({ data, label, unit }: { data: Diagnostics['points']; label: string; unit: string }) {
  const p = usePalette()
  const vals = data.flatMap((d) => [d.actual, d.predicted])
  const lo = Math.min(...vals)
  const hi = Math.max(...vals)
  const u = unit ? ` [${unit}]` : ''
  return (
    <ChartFrame title={`Actual vs predicted · ${label}`} caption="Held-out test set. Points on the dashed line are perfect predictions.">
      <ScatterChart margin={{ top: 8, right: 12, bottom: 20, left: 12 }}>
        <CartesianGrid strokeDasharray="0" />
        <XAxis type="number" dataKey="actual" domain={[lo, hi]} tickFormatter={(v) => num(v, 2)} label={axisLabel(`Actual${u}`, 'x')} />
        <YAxis type="number" dataKey="predicted" domain={[lo, hi]} tickFormatter={(v) => num(v, 2)} label={axisLabel(`Predicted${u}`, 'y')} width={64} />
        <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke={p.muted} strokeDasharray="5 4" strokeWidth={1.5} />
        <Tooltip
          cursor={{ strokeDasharray: '3 3' }}
          content={({ payload }) =>
            payload?.[0] ? <TooltipBox lines={[`actual ${num(payload[0].payload.actual)}${unit ? ' ' + unit : ''}`, `predicted ${num(payload[0].payload.predicted)}${unit ? ' ' + unit : ''}`]} /> : null
          }
        />
        <Scatter data={data} fill={p.series[0]} fillOpacity={0.7} isAnimationActive={false} shape="circle" />
      </ScatterChart>
    </ChartFrame>
  )
}

export function Residuals({ data, label, unit }: { data: Diagnostics['points']; label: string; unit: string }) {
  const p = usePalette()
  const u = unit ? ` [${unit}]` : ''
  return (
    <ChartFrame title={`Residuals · ${label}`} caption="Actual minus predicted. An even scatter around zero means no systematic bias.">
      <ScatterChart margin={{ top: 8, right: 12, bottom: 20, left: 12 }}>
        <CartesianGrid />
        <XAxis type="number" dataKey="predicted" tickFormatter={(v) => num(v, 2)} domain={['auto', 'auto']} label={axisLabel(`Predicted${u}`, 'x')} />
        <YAxis type="number" dataKey="residual" tickFormatter={(v) => num(v, 2)} label={axisLabel(`Residual${u}`, 'y')} width={64} />
        <ReferenceLine y={0} stroke={p.muted} strokeDasharray="5 4" strokeWidth={1.5} />
        <Tooltip content={({ payload }) => (payload?.[0] ? <TooltipBox lines={[`residual ${num(payload[0].payload.residual)}${unit ? ' ' + unit : ''}`]} /> : null)} />
        <Scatter data={data} fill={p.series[1]} fillOpacity={0.7} isAnimationActive={false} />
      </ScatterChart>
    </ChartFrame>
  )
}

export function ImportanceBars({ data, title, caption, withError }: { data: { short: string; mean: number; std?: number }[]; title: string; caption: string; withError?: boolean }) {
  const p = usePalette()
  const rows = [...data].sort((a, b) => b.mean - a.mean)
  return (
    <ChartFrame title={title} caption={caption} height={Math.max(180, rows.length * 30 + 50)}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 24, bottom: 16, left: 8 }}>
        <CartesianGrid horizontal={false} />
        <XAxis type="number" tickFormatter={(v) => num(v, 2)} label={axisLabel('Importance (drop in test R²)', 'x')} />
        <YAxis type="category" dataKey="short" width={132} tickLine={false} axisLine={false} />
        <Tooltip cursor={{ fill: 'transparent' }} content={({ payload }) => (payload?.[0] ? <TooltipBox lines={[`${payload[0].payload.short}: ${num(payload[0].payload.mean)}${payload[0].payload.std !== undefined ? ` ± ${num(payload[0].payload.std)}` : ''}`]} /> : null)} />
        <Bar dataKey="mean" fill={p.series[0]} radius={[0, 4, 4, 0]} barSize={14} isAnimationActive={false}>
          {withError && <ErrorBar dataKey="std" width={4} stroke={p.muted} direction="x" />}
        </Bar>
      </BarChart>
    </ChartFrame>
  )
}

/** Two-line category tick: dataset on the first line, target on the second (no clipping, no rotation). */
function GroupTick(props: { x?: number; y?: number; payload?: { value: string } }) {
  const [dataset, target] = String(props.payload?.value ?? '').split(' · ')
  return (
    <g transform={`translate(${props.x ?? 0},${(props.y ?? 0) + 6})`}>
      <text textAnchor="middle" className="recharts-cartesian-axis-tick-value" fontSize={11}>
        <tspan x={0} dy="0.8em" fontWeight={600}>{dataset}</tspan>
        <tspan x={0} dy="1.3em">{target}</tspan>
      </text>
    </g>
  )
}

const MODEL_ORDER = ['random_forest', 'gradient_boosting', 'extra_trees', 'dummy']

export function ModelComparison({ rows }: { rows: (Experiment & { dataset_label: string })[] }) {
  const p = usePalette()
  const models = MODEL_ORDER.filter((m) => rows.some((r) => r.model === m))
  const groups = new Map<string, Record<string, number | string>>()
  for (const r of rows) {
    if (r.variant !== 'default') continue
    const key = `${r.dataset_label} · ${r.target_short}`
    const g = groups.get(key) ?? { group: key }
    g[r.model] = r.r2 ?? 0
    groups.set(key, g)
  }
  const label = (m: string) => rows.find((r) => r.model === m)?.model_label ?? m
  const lowest = Math.min(0, ...rows.map((r) => r.r2 ?? 0))
  const yMin = lowest < -0.02 ? Math.floor(lowest * 4) / 4 : 0
  const yTicks = [yMin, 0, 0.25, 0.5, 0.75, 1].filter((t, i, a) => t >= yMin && a.indexOf(t) === i)
  const color = (m: string, i: number) => (m === 'dummy' ? p.muted : p.series[i % p.series.length])
  return (
    <ChartFrame
      title="Test R² by dataset and target"
      caption="Each group is one dataset and target; R² is unitless (1 = perfect, 0 = no better than the mean). Datasets describe different antennas, so bars across datasets are not like-for-like."
      height={320}
    >
      <BarChart data={[...groups.values()]} margin={{ top: 8, right: 8, bottom: 8, left: 8 }} barGap={2} barCategoryGap="22%">
        <CartesianGrid vertical={false} />
        <XAxis dataKey="group" interval={0} height={44} tick={<GroupTick />} tickLine={false} />
        <YAxis domain={[yMin, 1]} ticks={yTicks} tickFormatter={(v) => num(v, 2)} label={axisLabel('Test R²', 'y')} width={56} />
        <Tooltip cursor={{ fill: 'transparent' }} content={({ payload, label: l }) => (payload?.length ? <TooltipBox lines={[String(l), ...payload.map((x) => `${label(String(x.dataKey))}: ${num(Number(x.value))}`)]} /> : null)} />
        <Legend verticalAlign="top" height={30} formatter={(v) => label(String(v))} />
        <ReferenceLine y={0} stroke={p.muted} />
        {models.map((m, i) => (
          <Bar key={m} dataKey={m} fill={color(m, i)} radius={[4, 4, 0, 0]} maxBarSize={22} isAnimationActive={false} />
        ))}
      </BarChart>
    </ChartFrame>
  )
}

export function Convergence({ data, unitNote }: { data: { evaluation: number; best: number }[]; unitNote: string }) {
  const p = usePalette()
  return (
    <ChartFrame title="Differential Evolution convergence" caption={`Best objective found so far (${unitNote}; lower is better). Steps up front are constraint penalties.`} height={220}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 20, left: 12 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="evaluation" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(v) => num(v, 0)} label={axisLabel('Surrogate evaluations', 'x')} />
        <YAxis dataKey="best" tickFormatter={(v) => num(v, 2)} domain={['auto', 'auto']} width={64} label={axisLabel('Best objective', 'y')} />
        <Tooltip content={({ payload }) => (payload?.[0] ? <TooltipBox lines={[`evaluation ${payload[0].payload.evaluation}`, `best ${num(payload[0].payload.best)}`]} /> : null)} />
        <Line type="stepAfter" dataKey="best" stroke={p.series[0]} strokeWidth={2} dot={false} isAnimationActive={false} />
      </LineChart>
    </ChartFrame>
  )
}
