import * as React from 'react'
import { cn } from '@/lib/utils'

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement> & { invalid?: boolean }>(
  ({ className, invalid, ...props }, ref) => (
    <input
      ref={ref}
      aria-invalid={invalid || undefined}
      className={cn(
        'tabular h-10 w-full rounded-[var(--radius-control)] border border-hairline-strong bg-surface px-3 text-[0.9375rem] text-ink transition-[border-color,box-shadow] duration-150 placeholder:text-ink-3 focus-visible:border-accent focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-accent/15 disabled:opacity-50',
        invalid && 'border-danger focus-visible:border-danger focus-visible:ring-danger/15',
        className,
      )}
      {...props}
    />
  ),
)
Input.displayName = 'Input'
