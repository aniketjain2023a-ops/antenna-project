import { useQuery } from '@tanstack/react-query'
import { Download, History, LineChart as LineIcon } from 'lucide-react'
import * as React from 'react'
import { Link } from 'react-router-dom'
import { ActualVsPredicted, ImportanceBars, ModelComparison, Residuals } from '@/components/charts'
import { EmptyState, Notice, PageHeader, SectionTitle } from '@/components/common'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Segmented } from '@/components/ui/segmented'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TBody, TD, TH, THead, TR } from '@/components/ui/table'
import { api, type ResultsSummary } from '@/lib/api'
import { cn, num, signed } from '@/lib/utils'

type Metric = ResultsSummary['metrics'][number]

function MetricsTable({ rows }: { rows: Metric[] }) {
  const sorted = [...rows].sort(
    (a, b) => a.dataset_label.localeCompare(b.dataset_label) || a.target_short.localeCompare(b.target_short) || (b.r2 ?? -9) - (a.r2 ?? -9),
  )
  return (
    <Table>
      <THead>
        <TR>
          <TH>Dataset</TH>
          <TH>Target</TH>
          <TH>Model</TH>
          <TH className="text-right">Test R²</TH>
          <TH className="text-right">MAE</TH>
          <TH className="text-right">RMSE</TH>
          <TH className="text-right">CV R²</TH>
          <TH className="text-right">Train / test</TH>
        </TR>
      </THead>
      <TBody>
        {sorted.map((e) => (
          <TR key={e.id}>
            <TD className="text-ink-2">{e.dataset_label}</TD>
            <TD className="text-ink-2">{e.target_short}</TD>
            <TD>
              <span className="inline-flex items-center gap-2">
                {e.model_label}
                {e.variant !== 'default' && <Badge>{e.variant}</Badge>}
                {e.active && <Badge tone="accent">active</Badge>}
              </span>
            </TD>
            <TD className="text-right font-medium">{num(e.r2, 3)}</TD>
            <TD className="text-right">{num(e.mae, 3)} <span className="text-ink-3">{e.unit}</span></TD>
            <TD className="text-right">{num(e.rmse, 3)} <span className="text-ink-3">{e.unit}</span></TD>
            <TD className="text-right">{num(e.cv_r2_mean, 3)}</TD>
            <TD className="text-right text-ink-2">{e.n_train} / {e.n_test}</TD>
          </TR>
        ))}
      </TBody>
    </Table>
  )
}

function DiagnosticsCard({ metrics }: { metrics: Metric[] }) {
  const options = metrics.filter((m) => m.model !== 'dummy')
  const [expId, setExpId] = React.useState<string>(() => String(options.find((m) => m.active)?.id ?? options[0]?.id ?? ''))
  const diag = useQuery({ queryKey: ['diagnostics', expId], queryFn: () => api.diagnostics(Number(expId)), enabled: !!expId })
  const chosen = options.find((m) => String(m.id) === expId)
  if (!options.length) return null
  return (
    <Card>
      <CardContent className="space-y-8 pt-6">
        <div className="max-w-xl">
          <Select value={expId} onValueChange={setExpId}>
            <SelectTrigger aria-label="Model to inspect">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {options.map((m) => (
                <SelectItem key={m.id} value={String(m.id)}>
                  {m.dataset_label} · {m.target_short} · {m.model_label}
                  {m.active ? ' (active)' : ''}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        {diag.isLoading && <Skeleton className="h-72" />}
        {diag.error && <Notice tone="danger">{(diag.error as Error).message}</Notice>}
        {diag.data && chosen && (
          <div className="animate-rise space-y-10">
            <div className="grid gap-10 lg:grid-cols-2">
              <ActualVsPredicted data={diag.data.points} label={`${chosen.target_short} (${chosen.dataset_label})`} unit={chosen.unit} />
              <Residuals data={diag.data.points} label={`${chosen.target_short} (${chosen.dataset_label})`} unit={chosen.unit} />
            </div>
            <div className="grid gap-10 lg:grid-cols-2">
              {diag.data.importance.permutation && (
                <ImportanceBars
                  data={diag.data.importance.permutation}
                  withError
                  title={`Permutation importance · ${chosen.target_short} (${chosen.dataset_label})`}
                  caption="Drop in test R² when one input is shuffled (mean ± std over 10 repeats). It shows what the model relies on — an association, not proof of physical cause."
                />
              )}
              {diag.data.importance.impurity && (
                <ImportanceBars
                  data={diag.data.importance.impurity}
                  title={`Impurity importance · ${chosen.target_short} (${chosen.dataset_label})`}
                  caption="From the trees' training splits (sums to 1). Can be biased; read it together with permutation importance."
                />
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function RunsTable({ runs }: { runs: ResultsSummary['optimization_runs'] }) {
  return (
    <Table>
      <THead>
        <TR>
          <TH>Run</TH>
          <TH>Dataset</TH>
          <TH>Objective</TH>
          <TH className="text-right">Baseline</TH>
          <TH className="text-right">Candidate</TH>
          <TH className="text-right">Change</TH>
          <TH>Constraint</TH>
          <TH className="text-right">Evaluations</TH>
          <TH>When</TH>
        </TR>
      </THead>
      <TBody>
        {runs.map((r) => {
          const d = r.baseline_value !== null && r.candidate_value !== null ? r.candidate_value - r.baseline_value : null
          const good = d === null || !r.direction ? null : r.direction === 'minimize' ? d < 0 : d > 0
          return (
            <TR key={r.id}>
              <TD>#{r.id}</TD>
              <TD className="text-ink-2">{r.dataset_label}</TD>
              <TD>
                {r.direction === 'maximize' ? 'Max' : r.direction === 'minimize' ? 'Min' : ''} {r.objective_short}
              </TD>
              <TD className="text-right">{num(r.baseline_value, 3)} <span className="text-ink-3">{r.unit}</span></TD>
              <TD className="text-right font-medium">{num(r.candidate_value, 3)} <span className="text-ink-3">{r.unit}</span></TD>
              <TD className={cn('text-right font-medium', good === true ? 'text-success' : good === false ? 'text-danger' : 'text-ink-3')}>{signed(d, r.unit, 3)}</TD>
              <TD>
                {!r.constrained ? <Badge>none</Badge> : r.feasible ? <Badge tone="success">met</Badge> : <Badge tone="danger">violated</Badge>}
              </TD>
              <TD className="text-right text-ink-2">{r.evaluations?.toLocaleString()}</TD>
              <TD className="text-ink-3">{r.created_at}</TD>
            </TR>
          )
        })}
      </TBody>
    </Table>
  )
}

const EXPORTS: { table: 'datasets' | 'experiments' | 'predictions' | 'optimizations'; label: string }[] = [
  { table: 'experiments', label: 'Model experiments' },
  { table: 'optimizations', label: 'Optimization runs' },
  { table: 'predictions', label: 'Saved predictions' },
  { table: 'datasets', label: 'Datasets & provenance' },
]

export function ResultsPage() {
  const q = useQuery({ queryKey: ['results'], queryFn: api.results })
  const [filter, setFilter] = React.useState('all')
  const datasets = [...new Set(q.data?.metrics.map((m) => m.dataset_label) ?? [])]
  const metrics = (q.data?.metrics ?? []).filter((m) => filter === 'all' || m.dataset_label === filter)
  const runs = (q.data?.optimization_runs ?? []).filter((r) => filter === 'all' || r.dataset_label === filter)
  return (
    <>
      <PageHeader eyebrow="Results" title="What the models learned.">
        Held-out test metrics for every model, what each model relies on, and every optimization run — each labelled with the dataset it
        came from. Everything is stored in SQLite and can be downloaded.
      </PageHeader>
      {q.error && <Notice tone="danger" title="Backend unavailable.">{(q.error as Error).message}</Notice>}
      {q.isLoading && <Skeleton className="h-96 rounded-[var(--radius-card)]" />}
      {q.data && q.data.metrics.length === 0 && (
        <EmptyState icon={<LineIcon />} title="No models trained yet" action={<Button asChild variant="secondary"><Link to="/">Go to Home</Link></Button>}>
          Train models for a dataset on the Home page; their test metrics appear here.
        </EmptyState>
      )}
      {q.data && q.data.metrics.length > 0 && (
        <>
          <div className="mb-8 flex flex-wrap items-center gap-3">
            <Segmented label="Dataset filter" value={filter} onChange={setFilter} options={[{ value: 'all', label: 'All datasets' }, ...datasets.map((d) => ({ value: d, label: d }))]} />
            <span className="text-caption text-ink-3">
              {q.data.counts.experiments} experiments · {q.data.counts.optimizations} optimization runs · {q.data.counts.predictions} saved predictions
            </span>
          </div>

          <section>
            <SectionTitle title="Model performance" description="Latest run of each model on its dataset's held-out test set. MAE and RMSE are in each target's own unit and cannot be compared across targets." />
            <Card>
              <CardContent className="space-y-6 pt-6">
                <ModelComparison rows={metrics} />
                <MetricsTable rows={metrics} />
                <p className="text-caption text-ink-2">
                  No model was tested on another dataset: their inputs and targets differ, so cross-dataset evaluation is not meaningful. See DATASET_COMPARISON.md.
                </p>
              </CardContent>
            </Card>
          </section>

          <section className="mt-16">
            <SectionTitle title="Model diagnostics" description="Choose one trained model to see its test-set predictions and which inputs it relies on." />
            <DiagnosticsCard key={filter} metrics={metrics} />
          </section>

          <section className="mt-16">
            <SectionTitle title="Optimization history" description="Every Differential Evolution run, with the objective's predicted value for the baseline and the ML-selected candidate." />
            <Card>
              <CardContent className="pt-6">
                {runs.length ? (
                  <RunsTable runs={runs} />
                ) : (
                  <EmptyState icon={<History />} title="No optimization runs yet" action={<Button asChild variant="secondary"><Link to="/design">Go to Design & Optimize</Link></Button>}>
                    Runs are saved automatically when you optimize a design.
                  </EmptyState>
                )}
              </CardContent>
            </Card>
          </section>
        </>
      )}

      <section className="mt-16">
        <SectionTitle title="Downloads" description="CSV exports of every record in the database (all datasets)." />
        <div className="flex flex-wrap gap-2">
          {EXPORTS.map((e) => (
            <Button key={e.table} variant="secondary" asChild>
              <a href={api.exportUrl(e.table)} download>
                <Download /> {e.label}
              </a>
            </Button>
          ))}
        </div>
      </section>
    </>
  )
}
