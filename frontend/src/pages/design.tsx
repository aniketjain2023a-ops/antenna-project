import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Bookmark, Loader2, Play, RotateCcw, SlidersHorizontal } from 'lucide-react'
import * as React from 'react'
import { Link } from 'react-router-dom'
import { toast } from 'sonner'
import { Convergence } from '@/components/charts'
import { DataNotice, EmptyState, Notice, PageHeader, SectionTitle } from '@/components/common'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Segmented } from '@/components/ui/segmented'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Table, TBody, TD, TH, THead, TR } from '@/components/ui/table'
import { Hint } from '@/components/ui/tooltip'
import { api, type Dataset, type Design, type Direction, type Feature, type OptimizeResponse, type Target } from '@/lib/api'
import { useDatasets } from '@/lib/dataset-context'
import { cn, num, signed, withUnit } from '@/lib/utils'

/* ----------------------------------------------------------------------------- helpers */
const fixed4 = (v: number) => String(Number(v.toFixed(4)))
const inward = (lo: number, hi: number): [string, string] => [fixed4(Math.ceil(lo * 1e4) / 1e4), fixed4(Math.floor(hi * 1e4) / 1e4)]
const parse = (s: string) => (s.trim() === '' ? NaN : Number(s))

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = React.useState(value)
  React.useEffect(() => {
    const id = window.setTimeout(() => setV(value), ms)
    return () => window.clearTimeout(id)
  }, [value, ms])
  return v
}

function fieldState(f: Feature, raw: string): { error?: string; warning?: string } {
  const v = parse(raw)
  if (!Number.isFinite(v)) return { error: 'Enter a number.' }
  if (v < 0) return { error: 'Must not be negative.' }
  if (v < f.min || v > f.max) return { warning: `Outside the training range ${num(f.min)} – ${num(f.max)}: extrapolation.` }
  return {}
}

const orderTargets = (ts: Target[]) => [...ts].sort((a, b) => Number(!a.name.startsWith('S11')) - Number(!b.name.startsWith('S11')))

function better(direction: Direction, delta: number): boolean | null {
  if (!direction || Math.abs(delta) < 1e-12) return null
  return direction === 'minimize' ? delta < 0 : delta > 0
}

/* ------------------------------------------------------------------------ prediction */
function ParameterForm({ ds, values, setValues }: { ds: Dataset; values: Record<string, string>; setValues: React.Dispatch<React.SetStateAction<Record<string, string>>> }) {
  const groups = React.useMemo(() => {
    const g = new Map<string, Feature[]>()
    for (const f of ds.features) g.set(f.group, [...(g.get(f.group) ?? []), f])
    return [...g.entries()]
  }, [ds.features])
  const reset = () => setValues(Object.fromEntries(ds.features.map((f) => [f.name, fixed4(f.median)])))
  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle>Antenna parameters</CardTitle>
            <CardDescription>{ds.frequency_note}</CardDescription>
          </div>
          <Button variant="ghost" size="sm" onClick={reset} title="Reset every parameter to the median training design">
            <RotateCcw /> Median
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {groups.map(([group, feats]) => (
          <fieldset key={group}>
            {groups.length > 1 && <legend className="mb-3 text-micro uppercase text-ink-3">{group}</legend>}
            <div className="grid grid-cols-1 gap-x-4 gap-y-4 sm:grid-cols-2">
              {feats.map((f) => {
                const st = fieldState(f, values[f.name] ?? '')
                const id = `p-${f.name}`
                return (
                  <div key={f.name} className="min-w-0">
                    <div className="mb-1.5 flex items-center gap-1.5">
                      <Label htmlFor={id}>
                        {f.short}
                        {f.unit && <span className="font-normal text-ink-3"> ({f.unit})</span>}
                      </Label>
                      <Hint>{f.description}</Hint>
                    </div>
                    <Input
                      id={id}
                      inputMode="decimal"
                      value={values[f.name] ?? ''}
                      invalid={!!st.error}
                      onChange={(e) => setValues((v) => ({ ...v, [f.name]: e.target.value }))}
                    />
                    <p className={cn('mt-1 text-[0.75rem]', st.error ? 'text-danger' : st.warning ? 'text-warning' : 'text-ink-3')}>
                      {st.error ?? st.warning ?? `Valid ${num(f.min)} – ${num(f.max)}`}
                    </p>
                  </div>
                )
              })}
            </div>
          </fieldset>
        ))}
      </CardContent>
    </Card>
  )
}

function PredictionPanel({ ds, values, valid }: { ds: Dataset; values: Record<string, string>; valid: boolean }) {
  const params = React.useMemo(() => Object.fromEntries(Object.entries(values).map(([k, v]) => [k, parse(v)])), [values])
  const debounced = useDebounced(params, 250)
  const pending = JSON.stringify(debounced) !== JSON.stringify(params)
  const qc = useQueryClient()
  const pred = useQuery({
    queryKey: ['predict', ds.id, debounced],
    queryFn: () => api.predict(ds.id, debounced),
    enabled: valid,
    placeholderData: keepPreviousData,
    retry: false,
  })
  const save = useMutation({
    mutationFn: () => api.predict(ds.id, params, true),
    onSuccess: (r) => {
      toast.success(`Saved as prediction #${r.saved_id}`)
      qc.invalidateQueries({ queryKey: ['results'] })
    },
    onError: (e: Error) => toast.error('Could not save', { description: e.message }),
  })
  const r = pred.data
  const ordered = r ? [...r.predictions].sort((a, b) => Number(!a.target.startsWith('S11')) - Number(!b.target.startsWith('S11'))) : []
  const partial = ordered.filter((p) => p.partial_training)
  return (
    <Card className="lg:sticky lg:top-20">
      <CardHeader>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle>Performance prediction</CardTitle>
          <div className="flex items-center gap-1.5" aria-live="polite">
            {!valid ? (
              <Badge tone="danger">Fix the inputs</Badge>
            ) : pending || pred.isFetching ? (
              <Badge>
                <Loader2 className="animate-spin" /> Updating
              </Badge>
            ) : pred.error ? (
              <Badge tone="danger">Error</Badge>
            ) : r?.extrapolation ? (
              <Badge tone="warning">Extrapolation · unreliable</Badge>
            ) : r ? (
              <Badge tone="success">Within training range</Badge>
            ) : null}
          </div>
        </div>
        <CardDescription>Updates as you type. An ML estimate of the {ds.type === 'SIMULATION' ? 'simulated' : 'modelled'} response.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {pred.error && <Notice tone="danger">{(pred.error as Error).message}</Notice>}
        {!r && !pred.error && <p className="text-caption text-ink-2">Waiting for valid parameters…</p>}
        {r && (
          <div className={cn('grid grid-cols-2 gap-x-4 gap-y-5 transition-opacity duration-150', (pending || pred.isFetching) && 'opacity-60')}>
            {ordered.map((p) => (
              <div key={p.target} className="min-w-0">
                <div className="flex items-center gap-1.5 text-micro uppercase text-ink-3">
                  {p.short}
                  {p.partial_training && <Hint>Trained only on the {p.n_samples_used?.toLocaleString()} designs where this quantity is defined inside the simulated band. For designs without an in-band resonance the value is not meaningful.</Hint>}
                </div>
                <div className="text-figure mt-1 text-ink">{withUnit(p.value, p.unit, 3)}</div>
                <div className="mt-0.5 text-caption text-ink-2">
                  {p.uncertainty !== null ? `± ${num(p.uncertainty, 2)} ${p.unit} (approx. 1σ)` : 'No uncertainty estimate'}
                </div>
              </div>
            ))}
          </div>
        )}
        {r && (
          <Table>
            <THead>
              <TR>
                <TH>Target</TH>
                <TH>Model</TH>
                <TH className="text-right">Test R²</TH>
                <TH className="text-right">Test RMSE</TH>
              </TR>
            </THead>
            <TBody>
              {ordered.map((p) => (
                <TR key={p.target}>
                  <TD className="text-ink-2">{p.short}</TD>
                  <TD>{p.model_label}</TD>
                  <TD className="text-right">{num(p.r2, 3)}</TD>
                  <TD className="text-right">{withUnit(p.rmse, p.unit, 3)}</TD>
                </TR>
              ))}
            </TBody>
          </Table>
        )}
        {partial.length > 0 && (
          <p className="text-caption text-ink-2">
            {partial.map((p) => p.short).join(' and ')} {partial.length > 1 ? 'are' : 'is'} only meaningful for designs that resonate inside the simulated band.
          </p>
        )}
        <p className="text-caption text-ink-3">
          Uncertainty is an approximate model indicator (spread of the forest's trees), not a calibrated confidence interval.
        </p>
        <Button variant="secondary" size="sm" onClick={() => save.mutate()} disabled={!valid || !r || pending || save.isPending}>
          <Bookmark /> Save prediction
        </Button>
      </CardContent>
    </Card>
  )
}

/* ---------------------------------------------------------------------- optimization */
interface OptState {
  objective: string
  constraintOn: boolean
  cTarget: string
  cOp: '<=' | '>='
  cLimit: string
  free: string[]
  bounds: Record<string, [string, string]>
  baseline: 'current' | 'median'
  budget: string
  seed: string
}

function initialOpt(ds: Dataset): OptState {
  const objTargets = orderTargets(ds.targets).filter((t) => t.direction)
  const objective = ds.targets.some((t) => t.name === 'Gain_dBi') ? 'Gain_dBi' : objTargets[0]?.name ?? ''
  const hasS11 = ds.targets.some((t) => t.name === 'S11_dB')
  const free = ds.features.filter((f) => f.optimize_default).map((f) => f.name)
  return {
    objective,
    constraintOn: objective === 'Gain_dBi' && hasS11,
    cTarget: hasS11 ? 'S11_dB' : ds.targets.find((t) => t.name !== objective)?.name ?? '',
    cOp: '<=',
    cLimit: '-10',
    free: free.length ? free : ds.features.map((f) => f.name),
    bounds: Object.fromEntries(ds.features.map((f) => [f.name, inward(f.min, f.max)])),
    baseline: 'current',
    budget: '1500',
    seed: '42',
  }
}

function DesignCard({ title, design, ds, free, objective, reference, tone }: { title: string; design: Design; ds: Dataset; free: string[]; objective: string; reference?: Design; tone?: React.ReactNode }) {
  const feats = ds.features.filter((f) => free.includes(f.name))
  const targets = orderTargets(ds.targets).filter((t) => design.predictions[t.name] !== undefined)
  return (
    <Card className={cn('animate-rise', reference && 'ring-1 ring-accent/30')}>
      <CardHeader>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle>{title}</CardTitle>
          {tone}
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
          {feats.map((f) => {
            const v = design.parameters[f.name]
            const d = reference ? v - reference.parameters[f.name] : null
            return (
              <div key={f.name} className="min-w-0">
                <div className="truncate text-micro uppercase text-ink-3">{f.short}</div>
                <div className="tabular mt-0.5 text-[1.125rem] font-semibold text-ink">{withUnit(v, f.unit, 4)}</div>
                {d !== null && <div className="tabular text-caption text-ink-3">{signed(d, f.unit, 3)}</div>}
              </div>
            )
          })}
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-4 border-t border-hairline pt-4">
          {targets.map((t) => {
            const v = design.predictions[t.name]
            const ref = reference?.predictions[t.name]
            const d = reference && v !== null && ref !== null && ref !== undefined ? v - ref : null
            const good = d !== null ? better(t.direction, d) : null
            return (
              <div key={t.name} className={cn('min-w-0', t.name === objective && 'rounded-xl bg-surface-2 p-2 -m-2')}>
                <div className="truncate text-micro uppercase text-ink-3">
                  {t.short}
                  {t.name === objective && ' · objective'}
                </div>
                <div className="tabular mt-0.5 text-[1.25rem] font-semibold text-ink">{withUnit(v, t.unit, 3)}</div>
                {d !== null && (
                  <div className={cn('tabular text-caption font-medium', good === true ? 'text-success' : good === false ? 'text-danger' : 'text-ink-3')}>
                    {signed(d, t.unit, 3)}
                    {good !== null && <span className="sr-only">{good ? ' (better)' : ' (worse)'}</span>}
                  </div>
                )}
                {design.uncertainty[t.name] !== null && design.uncertainty[t.name] !== undefined && (
                  <div className="text-[0.75rem] text-ink-3">± {num(design.uncertainty[t.name], 2)} {t.unit}</div>
                )}
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}

function OptimizationResult({ ds, res }: { ds: Dataset; res: OptimizeResponse }) {
  const obj = ds.targets.find((t) => t.name === res.objective)!
  const feasibility = (d: Design) =>
    !res.constraint ? <Badge>Unconstrained</Badge> : d.feasible ? <Badge tone="success">Meets constraint</Badge> : <Badge tone="danger">Violates constraint</Badge>
  const fixed = ds.features.filter((f) => !res.free_parameters.includes(f.name))
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <DesignCard title="Baseline design" design={res.baseline} ds={ds} free={res.free_parameters} objective={res.objective} tone={feasibility(res.baseline)} />
        <DesignCard title="ML-selected candidate" design={res.candidate} ds={ds} free={res.free_parameters} objective={res.objective} reference={res.baseline} tone={feasibility(res.candidate)} />
      </div>
      <Notice tone="warning" title="Not physically validated.">
        This candidate was chosen by an ML surrogate ({res.n_evals.toLocaleString()} evaluations in {num(res.runtime_s, 2)} s, saved as run #{res.run_id}). It
        has not been simulated in HFSS/CST or measured. Green and red mark predicted changes that are better or worse for that target.
      </Notice>
      {fixed.length > 0 && (
        <p className="text-caption text-ink-2">
          Held at the baseline: {fixed.map((f) => `${f.short} ${withUnit(res.baseline.parameters[f.name], f.unit, 4)}`).join(' · ')}
        </p>
      )}
      <Card>
        <CardContent className="space-y-8 pt-6">
          <Convergence data={res.history} unitNote={`${obj.direction === 'maximize' ? 'negative of ' : ''}${obj.short}${obj.unit ? ` in ${obj.unit}` : ''}, plus constraint penalty`} />
          {res.nearest_simulated && (
            <div>
              <p className="mb-1 text-[0.9375rem] font-semibold text-ink">Nearest simulated design</p>
              <p className="mb-3 text-caption text-ink-2">
                The closest design that was actually simulated (normalized parameter distance; 0 = identical). A sanity reference, not a validation of the candidate.
              </p>
              <Table>
                <THead>
                  <TR>
                    <TH>Design</TH>
                    <TH className="text-right">Row</TH>
                    <TH className="text-right">Distance</TH>
                    {res.nearest_simulated[0].values.map((v) => (
                      <TH key={v.target} className="text-right">
                        {v.short}: simulated / predicted
                      </TH>
                    ))}
                  </TR>
                </THead>
                <TBody>
                  {res.nearest_simulated.map((n) => (
                    <TR key={n.design}>
                      <TD>{n.design}</TD>
                      <TD className="text-right">{n.row}</TD>
                      <TD className="text-right">{num(n.distance, 3)}</TD>
                      {n.values.map((v) => (
                        <TD key={v.target} className="text-right">
                          {num(v.dataset, 3)} / {num(v.predicted, 3)} <span className="text-ink-3">{v.unit}</span>
                        </TD>
                      ))}
                    </TR>
                  ))}
                </TBody>
              </Table>
            </div>
          )}
          {res.synthetic_oracle && (
            <div>
              <p className="mb-1 text-[0.9375rem] font-semibold text-ink">Synthetic oracle check</p>
              <p className="mb-3 text-caption text-ink-2">Noise-free generating formula at each design (only possible for synthetic data).</p>
              <Table>
                <THead>
                  <TR>
                    <TH>Design</TH>
                    <TH className="text-right">Formula S11</TH>
                    <TH className="text-right">Formula gain</TH>
                  </TR>
                </THead>
                <TBody>
                  {res.synthetic_oracle.map((o) => (
                    <TR key={o.design}>
                      <TD>{o.design}</TD>
                      <TD className="text-right">{num(o.S11_dB, 3)} dB</TD>
                      <TD className="text-right">{num(o.Gain_dBi, 3)} dBi</TD>
                    </TR>
                  ))}
                </TBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function OptimizationPanel({ ds, values, valid }: { ds: Dataset; values: Record<string, string>; valid: boolean }) {
  const [s, setS] = React.useState<OptState>(() => initialOpt(ds))
  const qc = useQueryClient()
  const set = <K extends keyof OptState>(k: K, v: OptState[K]) => setS((o) => ({ ...o, [k]: v }))
  const objTargets = orderTargets(ds.targets).filter((t) => t.direction)
  const others = ds.targets.filter((t) => t.name !== s.objective)

  const problems: string[] = []
  if (s.free.length === 0) problems.push('Choose at least one parameter to vary.')
  for (const name of s.free) {
    const f = ds.features.find((x) => x.name === name)!
    const [lo, hi] = s.bounds[name].map(parse)
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) problems.push(`${f.short}: enter numeric bounds.`)
    else if (lo >= hi) problems.push(`${f.short}: the minimum must be below the maximum.`)
    else if (lo < f.min - 1e-9 || hi > f.max + 1e-9) problems.push(`${f.short}: bounds must stay inside the training range ${num(f.min)} – ${num(f.max)}.`)
  }
  if (s.constraintOn && !Number.isFinite(parse(s.cLimit))) problems.push('Enter a numeric constraint limit.')
  const budget = parse(s.budget)
  if (!(budget >= 100 && budget <= 20000)) problems.push('The evaluation budget must be between 100 and 20,000.')
  let baseline: Record<string, number> = {}
  if (s.baseline === 'current') {
    if (!valid) problems.push('The current parameters are invalid; fix them above or use the median design as the baseline.')
    baseline = Object.fromEntries(ds.features.map((f) => [f.name, parse(values[f.name])]))
    const outside = ds.features.filter((f) => baseline[f.name] < f.min || baseline[f.name] > f.max)
    if (valid && outside.length) problems.push(`The baseline is outside the training range (${outside.map((f) => f.short).join(', ')}); held parameters would be extrapolated.`)
  } else {
    baseline = Object.fromEntries(ds.features.map((f) => [f.name, f.median]))
  }

  const run = useMutation({
    mutationFn: () =>
      api.optimize(ds.id, {
        objective: s.objective,
        constraint: s.constraintOn && s.cTarget ? { target: s.cTarget, operator: s.cOp, limit: parse(s.cLimit) } : null,
        bounds: Object.fromEntries(s.free.map((n) => [n, s.bounds[n].map(parse) as [number, number]])),
        baseline,
        budget,
        seed: parse(s.seed) || 0,
      }),
    onSuccess: (r) => {
      toast.success(`Optimization finished · run #${r.run_id}`)
      qc.invalidateQueries({ queryKey: ['results'] })
    },
    onError: (e: Error) => toast.error('Optimization could not run', { description: e.message }),
  })
  const objective = ds.targets.find((t) => t.name === s.objective)

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="grid grid-cols-1 gap-8 pt-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.35fr)_minmax(0,0.8fr)]">
          {/* Objective */}
          <div className="space-y-4">
            <p className="text-headline text-ink">Objective</p>
            <div>
              <Label className="mb-1.5 block">Target</Label>
              <Select value={s.objective} onValueChange={(v) => setS((o) => ({ ...o, objective: v, cTarget: o.cTarget === v ? (ds.targets.find((t) => t.name !== v)?.name ?? '') : o.cTarget }))}>
                <SelectTrigger aria-label="Objective">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {objTargets.map((t) => (
                    <SelectItem key={t.name} value={t.name}>
                      {t.direction === 'minimize' ? 'Minimize' : 'Maximize'} {t.short}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {others.length > 0 && (
              <div className="space-y-3">
                <label className="flex items-center gap-2.5 text-caption font-medium text-ink">
                  <Switch checked={s.constraintOn} onCheckedChange={(v) => set('constraintOn', v)} aria-label="Add a constraint" />
                  Add a constraint
                </label>
                {s.constraintOn && (
                  <div className="animate-rise grid grid-cols-[minmax(0,1.4fr)_auto_minmax(0,1fr)] items-end gap-2">
                    <Select value={s.cTarget} onValueChange={(v) => set('cTarget', v)}>
                      <SelectTrigger aria-label="Constrained target">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {others.map((t) => (
                          <SelectItem key={t.name} value={t.name}>
                            {t.short}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Segmented label="Rule" value={s.cOp} onChange={(v) => set('cOp', v)} options={[{ value: '<=', label: '≤' }, { value: '>=', label: '≥' }]} />
                    <Input aria-label="Constraint limit" inputMode="decimal" value={s.cLimit} onChange={(e) => set('cLimit', e.target.value)} />
                  </div>
                )}
              </div>
            )}
            <p className="text-caption text-ink-2">
              {objective?.description}. Constraints are checked on predictions and enforced with a penalty.
            </p>
          </div>

          {/* Parameters and bounds */}
          <div className="space-y-4">
            <p className="text-headline text-ink">Parameters to vary</p>
            <div className="flex flex-wrap gap-2">
              {ds.features.map((f) => {
                const on = s.free.includes(f.name)
                return (
                  <label key={f.name} className={cn('pressable flex cursor-pointer items-center gap-2 rounded-full border px-3 py-1.5 text-caption', on ? 'border-accent bg-accent-soft text-accent-ink' : 'border-hairline-strong text-ink-2')}>
                    <Checkbox checked={on} onCheckedChange={(v) => set('free', v ? [...s.free, f.name] : s.free.filter((x) => x !== f.name))} aria-label={`Vary ${f.short}`} />
                    {f.short}
                  </label>
                )
              })}
            </div>
            <div className="space-y-2.5">
              {s.free.map((name) => {
                const f = ds.features.find((x) => x.name === name)!
                return (
                  <div key={name} className="grid grid-cols-[minmax(0,1fr)_6.5rem_6.5rem] items-center gap-2">
                    <span className="truncate text-caption text-ink-2">
                      {f.short}
                      {f.unit && ` (${f.unit})`}
                    </span>
                    <Input aria-label={`${f.short} minimum`} inputMode="decimal" value={s.bounds[name][0]} onChange={(e) => set('bounds', { ...s.bounds, [name]: [e.target.value, s.bounds[name][1]] })} />
                    <Input aria-label={`${f.short} maximum`} inputMode="decimal" value={s.bounds[name][1]} onChange={(e) => set('bounds', { ...s.bounds, [name]: [s.bounds[name][0], e.target.value] })} />
                  </div>
                )
              })}
            </div>
            <div>
              <Label className="mb-1.5 block">Baseline and held values</Label>
              <Segmented label="Baseline" value={s.baseline} onChange={(v) => set('baseline', v)} options={[{ value: 'current', label: 'Parameters above' }, { value: 'median', label: 'Median design' }]} />
            </div>
          </div>

          {/* Search */}
          <div className="space-y-4">
            <p className="text-headline text-ink">Differential Evolution</p>
            <div>
              <div className="mb-1.5 flex items-center gap-1.5">
                <Label htmlFor="budget">Evaluation budget</Label>
                <Hint>Maximum number of surrogate evaluations. SciPy differential_evolution with a Latin-hypercube start; no gradient polish.</Hint>
              </div>
              <Input id="budget" inputMode="numeric" value={s.budget} onChange={(e) => set('budget', e.target.value)} />
            </div>
            <div>
              <Label htmlFor="seed" className="mb-1.5 block">Random seed</Label>
              <Input id="seed" inputMode="numeric" value={s.seed} onChange={(e) => set('seed', e.target.value)} />
            </div>
          </div>

          <div className="space-y-3 lg:col-span-3">
            {problems.map((p) => (
              <Notice key={p} tone="danger">
                {p}
              </Notice>
            ))}
            <Button size="lg" onClick={() => run.mutate()} disabled={problems.length > 0 || run.isPending}>
              {run.isPending ? <Loader2 className="animate-spin" /> : <Play />}
              {run.isPending ? 'Optimizing…' : 'Run optimization'}
            </Button>
            {run.error && <Notice tone="danger" title="Optimization could not run.">{(run.error as Error).message}</Notice>}
          </div>
        </CardContent>
      </Card>

      <section aria-live="polite">
        {run.data ? (
          <OptimizationResult ds={ds} res={run.data} />
        ) : (
          <EmptyState icon={<SlidersHorizontal />} title="No optimization yet">
            Choose the objective, the parameters to vary and their bounds, then press Run optimization. The baseline and the ML-selected candidate will appear here side by side.
          </EmptyState>
        )}
      </section>
    </div>
  )
}

/* ------------------------------------------------------------------------------ page */
function Workspace({ ds }: { ds: Dataset }) {
  const [values, setValues] = React.useState<Record<string, string>>(() => Object.fromEntries(ds.features.map((f) => [f.name, fixed4(f.median)])))
  const valid = ds.features.every((f) => !fieldState(f, values[f.name] ?? '').error)
  return (
    <>
      <div className="mb-8">
        <DataNotice dataset={ds} />
      </div>
      <section>
        <SectionTitle title="Predict performance" description="Enter one design; the active model for each target predicts its performance." />
        <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
          <ParameterForm ds={ds} values={values} setValues={setValues} />
          <PredictionPanel ds={ds} values={values} valid={valid} />
        </div>
      </section>
      <section className="mt-16">
        <SectionTitle
          title="Optimize design"
          description="A separate operation: Differential Evolution searches within the bounds for the design that best meets the objective, calling the same models. Unselected parameters stay at the baseline."
        />
        <OptimizationPanel ds={ds} values={values} valid={valid} />
      </section>
    </>
  )
}

export function DesignPage() {
  const { data, isLoading, error, selected, select } = useDatasets()
  const ready = data?.datasets.filter((d) => d.ready) ?? []
  return (
    <>
      <PageHeader eyebrow="Design & Optimize" title="Explore a design.">
        Choose a dataset, predict the performance of one antenna design, then let Differential Evolution search for a better one. Each dataset has its own inputs and models; they are never mixed.
      </PageHeader>
      {error && <Notice tone="danger" title="Backend unavailable.">{error.message}</Notice>}
      {isLoading && <p className="text-caption text-ink-2">Loading datasets…</p>}
      {data && (
        <div className="mb-8 flex flex-wrap items-center gap-x-4 gap-y-2">
          {ready.length > 0 && selected && (
            <Segmented label="Dataset" value={String(selected.ready ? selected.id : ready[0].id)} onChange={(v) => select(Number(v))} options={ready.map((d) => ({ value: String(d.id), label: d.label }))} />
          )}
          {data.unavailable.map((u) => (
            <span key={u.key} className="text-caption text-ink-3">
              {u.label}: not available (pipeline incomplete)
            </span>
          ))}
        </div>
      )}
      {data && ready.length === 0 && (
        <EmptyState icon={<SlidersHorizontal />} title="No trained models yet" action={<Button asChild variant="secondary"><Link to="/">Go to Home</Link></Button>}>
          Prediction and optimization need trained models. Train them for a dataset on the Home page.
        </EmptyState>
      )}
      {selected && !selected.ready && ready.length > 0 && (
        <Notice>The dataset selected on Home has no trained models, so {ready[0].label} is shown. Train models on Home to use it here.</Notice>
      )}
      {(() => {
        const ds = selected?.ready ? selected : ready[0]
        return ds ? <Workspace key={ds.id} ds={ds} /> : null
      })()}
    </>
  )
}
