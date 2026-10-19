import { useQuery } from '@tanstack/react-query'
import * as React from 'react'
import { api, type Dataset, type DatasetsResponse } from '@/lib/api'

interface Ctx {
  data: DatasetsResponse | undefined
  isLoading: boolean
  error: Error | null
  selected: Dataset | undefined
  select: (id: number) => void
}

const DatasetContext = React.createContext<Ctx | null>(null)
const STORAGE_KEY = 'metaopt.dataset'

function readStored(): number | null {
  try {
    const v = window.localStorage.getItem(STORAGE_KEY)
    return v ? Number(v) : null
  } catch {
    return null
  }
}

/** Shares the selected dataset across the three pages (remembered per browser as a convenience). */
export function DatasetProvider({ children }: { children: React.ReactNode }) {
  const query = useQuery({ queryKey: ['datasets'], queryFn: api.datasets })
  const [selectedId, setSelectedId] = React.useState<number | null>(readStored)
  const datasets = query.data?.datasets ?? []
  const selected = datasets.find((d) => d.id === selectedId) ?? datasets.find((d) => d.ready) ?? datasets[0]
  const select = React.useCallback((id: number) => {
    setSelectedId(id)
    try {
      window.localStorage.setItem(STORAGE_KEY, String(id))
    } catch {
      /* storage unavailable: selection still works for this session */
    }
  }, [])
  return (
    <DatasetContext.Provider value={{ data: query.data, isLoading: query.isLoading, error: query.error, selected, select }}>
      {children}
    </DatasetContext.Provider>
  )
}

export function useDatasets(): Ctx {
  const ctx = React.useContext(DatasetContext)
  if (!ctx) throw new Error('useDatasets must be used inside DatasetProvider')
  return ctx
}
