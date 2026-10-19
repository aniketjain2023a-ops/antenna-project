import { NavLink, Outlet } from 'react-router-dom'
import { cn } from '@/lib/utils'

const NAV = [
  { to: '/', label: 'Home', end: true },
  { to: '/design', label: 'Design & Optimize', end: false },
  { to: '/results', label: 'Results', end: false },
]

function Mark() {
  return (
    <svg viewBox="0 0 32 32" className="size-7" aria-hidden>
      <rect width="32" height="32" rx="8" fill="var(--accent)" />
      <path d="M9 22l7-12 7 12" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="16" cy="10" r="2.2" fill="#fff" />
    </svg>
  )
}

export function AppShell() {
  return (
    <div className="min-h-dvh">
      {/* Translucent navigation material: content scrolls underneath (skill §12). */}
      <header className="material-nav sticky top-0 z-40">
        <nav className="mx-auto flex h-14 max-w-[76rem] items-center gap-4 px-4 sm:px-6" aria-label="Main">
          <NavLink to="/" className="flex shrink-0 items-center gap-2.5" aria-label="MetaOpt-AI home">
            <Mark />
            <span className="hidden text-[0.9375rem] font-semibold tracking-[-0.01em] sm:inline">MetaOpt-AI</span>
          </NavLink>
          <div className="ml-auto flex items-center gap-1 overflow-x-auto">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  cn(
                    'pressable whitespace-nowrap rounded-full px-3.5 py-1.5 text-caption font-medium',
                    isActive ? 'bg-ink text-canvas' : 'text-ink-2 hover:bg-surface-2 hover:text-ink',
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </div>
        </nav>
      </header>
      <main className="mx-auto max-w-[76rem] px-4 pb-24 pt-10 sm:px-6 sm:pt-14">
        <Outlet />
      </main>
      <footer className="mx-auto max-w-[76rem] px-4 pb-10 text-caption text-ink-3 sm:px-6">
        MetaOpt-AI · Aniket Jain · 23BEC0470 · ML predictions are not EM simulations or measurements.
      </footer>
    </div>
  )
}
