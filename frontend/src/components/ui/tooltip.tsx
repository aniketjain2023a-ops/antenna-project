import * as TooltipPrimitive from '@radix-ui/react-tooltip'
import { Info } from 'lucide-react'
import * as React from 'react'

export const TooltipProvider = TooltipPrimitive.Provider

/** Small "i" affordance that explains a technical term. */
export function Hint({ children, label = 'More information' }: { children: React.ReactNode; label?: string }) {
  return (
    <TooltipPrimitive.Root delayDuration={150}>
      <TooltipPrimitive.Trigger asChild>
        <button type="button" aria-label={label} className="inline-flex text-ink-3 hover:text-ink-2">
          <Info className="size-3.5" />
        </button>
      </TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content
          sideOffset={6}
          className="animate-rise z-50 max-w-72 rounded-lg bg-ink px-3 py-2 text-caption text-canvas shadow-lg"
        >
          {children}
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  )
}
