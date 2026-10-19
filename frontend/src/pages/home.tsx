import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowRight, Ban, BrainCircuit, CheckCircle2, Database, LineChart, Loader2, SlidersHorizontal, Target } from 'lucide-react'
import * as React from 'react'
import { Link } from 'react-router-dom'
import { toast } from 'sonner'
import { DataNotice, EmptyState, Notice, PageHeader, ProvenanceBadge, SectionTitle, Stat } from '@/components/common'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Disclosure } from '@/components/ui/collapsible'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TBody, TD, TH, THead, TR } from '@/components/ui/table'
import { Hint } from '@/components/ui/tooltip'
import { api, type Dataset, type Experiment } from '@/lib/api'
import { useDatasets } from '@/lib/dataset-context'
import { cn, num } from '@/lib/utils'

const STEPS = [
  { icon: <Database />, title: 'Dataset', text: 'Verified source, provenance and checksum' },
  { icon: <BrainCircuit />, title: 'Train', text: 'Dummy, Random Forest, Gradient Boosting' },
  { icon: <Target />, title: 'Predict', text: 'Performance of one design, with uncertainty' },
  { icon: <SlidersHorizontal />, title: 'Optimize', text: 'Differential Evolution within bounds' },
  { icon: <LineChart />, title: 'Compare', text: 'Metrics and runs across datasets' },
]

function Workflow() {
  return (
    <ol className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
      {STEPS.map((s, i) => (
        <li key={s.title} className="rounded-2xl border border-hairline bg-surface p-4">
          <div className="flex items-center gap-2 text-accent [&_svg]:size-4">
            {s.icon}
            <span className="text-micro uppercase text-ink-3">Step {i + 1}</span>
          </div>
          <p className="mt-2 text-[0.9375rem] font-semibold text-ink">{s.title}</p>
          <p className="mt-0.5 text-caption text-ink-2">{s.text}</p>
        </li>
      ))}
    </ol>
  )
}

function DatasetCard({ ds, selected, onSelect }: { ds: Dataset; selected: boolean; onSelect: () => void }) {
  const best = Object.values(ds.active_models)
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      className={cn(
        'pressable flex h-full flex-col rounded-[var(--radius-card)] border bg-surface p-5 text-left',
        selected ? 'border-accent ring-4 ring-accent/12' : 'border-hairline hover:border-hairline-strong',
      )}
    >
      <div className="flex w-full items-start justify-between gap-3">
        <p className="text-headline text-ink">{ds.label}</p>
        {selected && <CheckCircle2 className="size-5 shrink-0 text-accent" />}
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        <ProvenanceBadge type={ds.type} />
        <Badge>{ds.sample_count.toLocaleString()} samples</Badge>
        <Badge>{ds.features.length} inputs</Badge>
      </div>
      <p className="mt-3 line-clamp-2 text-caption text-ink-2">{ds.description || ds.name}</p>
      <div className="mt-4 w-full space-y-1.5 border-t border-hairline pt-3">
        {best.length === 0 ? (
          <p className="text-caption text-ink-3">No trained models yet</p>
        ) : (
          best.map((e) => (
            <div key={e.id} className="flex items-center justify-between gap-2 text-caption">
              <span className="truncate text-ink-2">{e.target_short}</span>
              <span className="tabular shrink-0 text-ink">
                R² {num(e.r2, 3)} <span className="text-ink-3">· {e.model_label}</span>
              </span>
            </div>
          ))
        )}
      </div>
    </button>
  )
}

function ExperimentTable({ experiments }: { experiments: Experiment[] }) {
  const latest = experiments.length ? experiments.filter((e) => e.run_group === experiments[0].run_group) : []
  const rows = [...latest].sort((a, b) => a.target.localeCompare(b.target) || (b.cv_r2_mean ?? -9) - (a.cv_r2_mean ?? -9))
  return (
    <Table>
      <THead>
        <TR>
          <TH>Target</TH>
          <TH>Model</TH>
          <TH className="text-right">
            <span className="inline-flex items-center gap-1">
              Test R² <Hint>Share of variance explained on the held-out test set. 1 = perfect; 0 = no better than predicting the mean.</Hint>
            </span>
          </TH>
          <TH className="text-right">MAE</TH>
          <TH className="text-right">RMSE</TH>
          <TH className="text-right">
            <span className="inline-flex items-center gap-1">
              CV R² <Hint>Mean 5-fold cross-validated R² on the training split. Used to choose the active model, so the test set never influences the choice.</Hint>
            </span>
          </TH>
        </TR>
      </THead>
      <TBody>
        {rows.map((e) => (
          <TR key={e.id} className={e.active ? 'bg-accent-soft/40' : undefined}>
            <TD className="text-ink-2">{e.target_short}</TD>
            <TD>
              <span className="inline-flex items-center gap-2">
                {e.model_label}
                {e.active && <Badge tone="accent">active</Badge>}
              </span>
            </TD>
            <TD className="text-right font-medium">{num(e.r2, 3)}</TD>
            <TD className="text-right">{num(e.mae, 3)} <span className="text-ink-3">{e.unit}</span></TD>
            <TD className="text-right">{num(e.rmse, 3)} <span className="text-ink-3">{e.unit}</span></TD>
            <TD className="text-right">{num(e.cv_r2_mean, 3)}</TD>
          </TR>
        ))}
      </TBody>
    </Table>
  )
}

function TrainingPanel({ ds, models, defaults }: { ds: Dataset; models: { key: string; label: string; description: string }[]; defaults: string[] }) {
  const qc = useQueryClient()
  const [chosen, setChosen] = React.useState<string[]>(defaults)
  const [elapsed, setElapsed] = React.useState(0)
  const experiments = useQuery({ queryKey: ['experiments', ds.id], queryFn: () => api.experiments(ds.id) })
  const train = useMutation({
    mutationFn: () => api.train(ds.id, chosen),
    onSuccess: (r) => {
      toast.success(`Trained ${r.experiments.length} models in ${r.seconds} s`, { description: 'Active models updated by cross-validated R².' })
      qc.invalidateQueries({ queryKey: ['experiments', ds.id] })
      qc.invalidateQueries({ queryKey: ['datasets'] })
      qc.invalidateQueries({ queryKey: ['results'] })
    },
    onError: (e: Error) => toast.error('Training failed', { description: e.message }),
  })
  React.useEffect(() => {
    if (!train.isPending) return
    const t0 = Date.now()
    const id = window.setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 500)
    return () => window.clearInterval(id)
  }, [train.isPending])

  return (
    <Card>
      <CardHeader>
        <CardTitle>Model training</CardTitle>
        <CardDescription>
          One model per target. 80/20 split with seed 42; 5-fold cross-validation on the training split chooses the active model.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <fieldset>
          <legend className="mb-2 text-caption font-medium text-ink">Models</legend>
          <div className="flex flex-wrap gap-2">
            {models.map((m) => {
              const on = chosen.includes(m.key)
              return (
                <label
                  key={m.key}
                  className={cn(
                    'pressable flex cursor-pointer items-center gap-2 rounded-full border px-3 py-1.5 text-caption',
                    on ? 'border-accent bg-accent-soft text-accent-ink' : 'border-hairline-strong text-ink-2',
                  )}
                  title={m.description}
                >
                  <Checkbox
                    checked={on}
                    onCheckedChange={(v) => setChosen((c) => (v ? [...c, m.key] : c.filter((k) => k !== m.key)))}
                    aria-label={m.label}
                  />
                  {m.label}
                </label>
              )
            })}
          </div>
        </fieldset>
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={() => train.mutate()} disabled={train.isPending || chosen.length === 0}>
            {train.isPending ? <Loader2 className="animate-spin" /> : <BrainCircuit />}
            {train.isPending ? 'Training…' : 'Train models'}
          </Button>
          <span className="text-caption text-ink-2" aria-live="polite">
            {train.isPending
              ? `Training ${chosen.length} model types × ${ds.targets.length} targets · ${elapsed}s (about ${ds.sample_count > 1000 ? '1–2 min' : '10 s'})`
              : chosen.length === 0
                ? 'Choose at least one model.'
                : `${ds.targets.length} targets: ${ds.targets.map((t) => t.short).join(', ')}`}
          </span>
        </div>
        {experiments.isLoading ? (
          <Skeleton className="h-40" />
        ) : experiments.data && experiments.data.experiments.length > 0 ? (
          <div className="animate-rise">
            <p className="mb-2 text-caption text-ink-2">Latest training run · metrics on the held-out test set</p>
            <ExperimentTable experiments={experiments.data.experiments} />
          </div>
        ) : (
          <EmptyState icon={<BrainCircuit />} title="No models trained yet">
            Choose the models above and press Train models. Metrics on the held-out test set appear here.
          </EmptyState>
        )}
      </CardContent>
    </Card>
  )
}

function DataCard({ ds }: { ds: Dataset }) {
  const preview = useQuery({ queryKey: ['preview', ds.id], queryFn: () => api.preview(ds.id) })
  return (
    <Card>
      <CardHeader>
        <CardTitle>Data</CardTitle>
        <CardDescription>{ds.name}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid grid-cols-3 gap-4">
          <Stat label="Samples" value={ds.quality.rows.toLocaleString()} />
          <Stat
            label="Missing"
            value={ds.quality.missing_values.toLocaleString()}
            hint={<Hint>Missing target values are never filled in. For METU-SIMPL they mark resonances outside the simulated 23–33 GHz band.</Hint>}
          />
          <Stat label="Duplicates" value={ds.quality.duplicate_rows} />
        </div>
        <div>
          <p className="mb-2 text-micro uppercase text-ink-3">Inputs</p>
          <div className="flex flex-wrap gap-1.5">
            {ds.features.map((f) => (
              <Badge key={f.name} tone="accent" title={f.description}>
                {f.short}
                {f.unit && <span className="opacity-70">· {f.unit}</span>}
              </Badge>
            ))}
          </div>
          <p className="mb-2 mt-4 text-micro uppercase text-ink-3">Targets</p>
          <div className="flex flex-wrap gap-1.5">
            {ds.targets.map((t) => (
              <Badge key={t.name} title={t.description}>
                {t.short}
                {t.unit && <span className="opacity-70">· {t.unit}</span>}
                {t.n_values < ds.sample_count && <span className="opacity-70">· {t.n_values.toLocaleString()} values</span>}
              </Badge>
            ))}
          </div>
        </div>
        <Disclosure title="Provenance and checksum">
          <p className="text-caption text-ink-2">{ds.provenance}</p>
          <p className="mt-2 break-all font-mono text-[0.75rem] text-ink-3">SHA-256 {ds.checksum}</p>
        </Disclosure>
        <Disclosure title="Preview first rows">
          {preview.data ? (
            <Table>
              <THead>
                <TR>
                  {preview.data.columns.map((c) => (
                    <TH key={c}>{c}</TH>
                  ))}
                </TR>
              </THead>
              <TBody>
                {preview.data.rows.map((r, i) => (
                  <TR key={i}>
                    {r.map((v, j) => (
                      <TD key={j}>{num(v, 3)}</TD>
                    ))}
                  </TR>
                ))}
              </TBody>
            </Table>
          ) : (
            <Skeleton className="h-24" />
          )}
        </Disclosure>
      </CardContent>
    </Card>
  )
}

export function HomePage() {
  const { data, isLoading, error, selected, select } = useDatasets()
  return (
    <>
      <PageHeader eyebrow="MetaOpt-AI · Aniket Jain · 23BEC0470" title="Antenna design, predicted and optimized.">
        Machine-learning surrogates trained on verified antenna data predict reflection (S11), resonance and bandwidth from
        geometry, then Differential Evolution searches for a better design. Every result is an ML prediction until it is
        simulated or measured.
      </PageHeader>
      <Workflow />

      <section className="mt-16">
        <SectionTitle title="Datasets" description="Each dataset keeps its own pipeline and models. A public source appears only after its download, preprocessing and training succeeded." />
        {error && <Notice tone="danger" title="Backend unavailable.">{error.message}</Notice>}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {isLoading && [0, 1, 2].map((i) => <Skeleton key={i} className="h-56 rounded-[var(--radius-card)]" />)}
          {data?.datasets.map((ds) => (
            <DatasetCard key={ds.id} ds={ds} selected={selected?.id === ds.id} onSelect={() => select(ds.id)} />
          ))}
          {data?.unavailable.map((u) => (
            <div key={u.key} className="flex flex-col rounded-[var(--radius-card)] border border-dashed border-hairline-strong p-5">
              <div className="flex items-center gap-2">
                <Ban className="size-4 text-ink-3" />
                <p className="text-headline text-ink-2">{u.label}</p>
              </div>
              <Badge className="mt-2 self-start">Not available</Badge>
              <p className="mt-3 text-caption text-ink-2">
                The pipeline did not complete, so no data or results from this source are used anywhere.
              </p>
              <p className="mt-2 break-words text-caption text-ink-3">{u.message}</p>
            </div>
          ))}
        </div>
      </section>

      {selected && (
        <section className="mt-16">
          <SectionTitle
            title={selected.label}
            description={selected.type_description}
            action={
              selected.ready && (
                <Button variant="secondary" asChild>
                  <Link to="/design">
                    Design & Optimize <ArrowRight />
                  </Link>
                </Button>
              )
            }
          />
          <div className="mb-4">
            <DataNotice dataset={selected} />
          </div>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
            <DataCard ds={selected} />
            {data && <TrainingPanel key={selected.id} ds={selected} models={data.models} defaults={data.default_models} />}
          </div>
        </section>
      )}
    </>
  )
}
