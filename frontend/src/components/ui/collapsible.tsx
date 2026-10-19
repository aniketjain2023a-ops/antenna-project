import * as CollapsiblePrimitive from '@radix-ui/react-collapsible'
import { ChevronRight } from 'lucide-react'
import * as React from 'react'

/** Disclosure: advanced or explanatory content one level deeper (simplicity, §16). */
export function Disclosure({ title, children, defaultOpen = false }: { title: string; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = React.useState(defaultOpen)
  return (
    <CollapsiblePrimitive.Root open={open} onOpenChange={setOpen}>
      <CollapsiblePrimitive.Trigger className="pressable group inline-flex items-center gap-1.5 text-caption font-medium text-ink-2 hover:text-ink">
        <ChevronRight className="size-4 transition-transform duration-200 ease-out group-data-[state=open]:rotate-90" />
        {title}
      </CollapsiblePrimitive.Trigger>
      <CollapsiblePrimitive.Content className="animate-rise pt-3">{children}</CollapsiblePrimitive.Content>
    </CollapsiblePrimitive.Root>
  )
}
