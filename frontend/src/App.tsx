import { lazy, Suspense } from 'react'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import { AppShell } from '@/components/app-shell'
import { Skeleton } from '@/components/ui/skeleton'

// Pages load on demand so the charting library is not part of the first download.
const HomePage = lazy(() => import('@/pages/home').then((m) => ({ default: m.HomePage })))
const DesignPage = lazy(() => import('@/pages/design').then((m) => ({ default: m.DesignPage })))
const ResultsPage = lazy(() => import('@/pages/results').then((m) => ({ default: m.ResultsPage })))

const page = (el: React.ReactNode) => <Suspense fallback={<Skeleton className="h-96 rounded-[var(--radius-card)]" />}>{el}</Suspense>

const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      { path: '/', element: page(<HomePage />) },
      { path: '/design', element: page(<DesignPage />) },
      { path: '/results', element: page(<ResultsPage />) },
      { path: '*', element: page(<HomePage />) },
    ],
  },
])

export default function App() {
  return <RouterProvider router={router} />
}
