---
name: animation-standards
description: Precise value tables for animation review — easing curves, duration budgets, spring configs, physicality rules, performance rules, a11y requirements. Reference cited by the critique-stage animation review variant.
---

# Animation Standards Reference

The precise values, curves, and rules behind the review. Cite these in findings instead of approximating.

## Should it animate? (frequency table)

| Frequency | Decision |
| --- | --- |
| 100+ times/day (keyboard shortcuts, command palette toggle) | No animation. Ever. |
| Tens of times/day (hover effects, list navigation) | Remove or drastically reduce |
| Occasional (modals, drawers, toasts) | Standard animation |
| Rare / first-time (onboarding, feedback, celebrations) | Can add delight |

**Never animate keyboard-initiated actions.** Raycast has no open/close animation — correct for something used hundreds of times a day.

Valid purposes: spatial consistency, state indication, explanation, feedback, preventing jarring change. "It looks cool" on a frequently-seen element is not valid.

## Easing

Decision order:
- Entering or exiting → **`ease-out`** (starts fast, feels responsive)
- Moving / morphing on screen → **`ease-in-out`**
- Hover / color change → **`ease`**
- Constant motion (marquee, progress) → **`linear`**
- Default → **`ease-out`**

**Never `ease-in` on UI.** It starts slow, delaying the exact moment the user is watching. `ease-out` at 200ms *feels* faster than `ease-in` at 200ms.

Built-in CSS easings are too weak. Use strong custom curves:

```css
--ease-out: cubic-bezier(0.23, 1, 0.32, 1);        /* strong ease-out for UI */
--ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);    /* strong ease-in-out */
--ease-drawer: cubic-bezier(0.32, 0.72, 0, 1);     /* iOS-like drawer curve */
```

## Duration

| Element | Duration |
| --- | --- |
| Button press feedback | 100–160ms |
| Tooltips, small popovers | 125–200ms |
| Dropdowns, selects | 150–250ms |
| Modals, drawers | 200–500ms |
| Marketing / explanatory | Can be longer |

**Rule: UI animations stay under 300ms.** A 180ms dropdown feels more responsive than a 400ms one.

## Physicality

- **Never `scale(0)`.** Start from `scale(0.9–0.97)` + `opacity: 0`.
- **Origin-aware popovers:** Scale from the trigger, not center:
  ```css
  .popover { transform-origin: var(--radix-popover-content-transform-origin); }
  ```
  **Modals are exempt** — they appear centered; keep `transform-origin: center`.
- **Button press feedback:** `transform: scale(0.97)` on `:active`, `transition: transform 160ms ease-out`.

## Springs

```js
// Apple-style (recommended — easier to reason about)
{ type: "spring", duration: 0.5, bounce: 0.2 }

// Traditional physics (more control)
{ type: "spring", mass: 1, stiffness: 100, damping: 10 }
```

Keep bounce subtle (0.1–0.3); reserve visible bounce for drag-to-dismiss and playful interactions. Springs maintain velocity when interrupted; keyframes restart from zero.

## Interruptibility

CSS **transitions** can be interrupted and retargeted mid-animation; **keyframes** restart from zero. For anything triggered rapidly, transitions are smoother.

```css
/* Interruptible — good for dynamic UI */
.toast { transition: transform 400ms ease; }

/* Not interruptible — avoid for dynamic UI */
@keyframes slideIn { from { transform: translateY(100%); } to { transform: translateY(0); } }
```

Use `@starting-style` for entry without JS:
```css
.toast {
  opacity: 1; transform: translateY(0);
  transition: opacity 400ms ease, transform 400ms ease;
  @starting-style { opacity: 0; transform: translateY(100%); }
}
```

## Asymmetric timing

Slow where the user is deciding, fast where the system responds:
```css
.overlay { transition: clip-path 200ms ease-out; }
.button:active .overlay { transition: clip-path 2s linear; }
```

## Performance

- **Only animate `transform` and `opacity`** — they skip layout/paint and run on the GPU.
- **Don't drive child transforms via a CSS variable on the parent** — it recalcs styles for all children. Set `transform` directly on the element.
- **Framer Motion shorthands are NOT hardware-accelerated.** `x`/`y`/`scale` run on the main thread via rAF and drop frames under load. Use the full transform string: `animate={{ transform: "translateX(100px)" }}`.
- **CSS animations beat JS under load** — they run off the main thread.

## Stagger

Stagger group entrances; 30–80ms between items. Longer delays feel slow. Stagger is decorative — never block interaction.

```css
.item { opacity: 0; transform: translateY(8px); animation: fadeIn 300ms ease-out forwards; }
.item:nth-child(2) { animation-delay: 50ms; }
.item:nth-child(3) { animation-delay: 100ms; }
@keyframes fadeIn { to { opacity: 1; transform: translateY(0); } }
```

## Accessibility

```css
@media (prefers-reduced-motion: reduce) {
  .element { animation: fade 0.2s ease; } /* keep opacity/color, drop movement */
}
@media (hover: hover) and (pointer: fine) {
  .element:hover { transform: scale(1.05); } /* gate hover motion — touch fires false hovers on tap */
}
```

Reduced motion means fewer and gentler animations, **not zero** — keep transitions that aid comprehension, remove movement/position changes.

## Gestures & drag

- **Momentum dismissal:** compute velocity (`Math.abs(distance)/elapsedMs`); dismiss if `> ~0.11`. A flick should be enough.
- **Damping at boundaries:** dragging past a natural edge moves less the further you go.
- **Pointer capture** once dragging starts.
- **Multi-touch protection:** ignore extra touch points.
- **Friction over hard stops** — allow over-drag with rising resistance.

## Masking imperfect crossfades

When a crossfade shows two overlapping states, add subtle `filter: blur(2px)` during the transition. Keep blur < 20px.

## Debugging

- **Slow motion:** bump duration 2–5× or use DevTools animation inspector.
- **Frame-by-frame:** Chrome DevTools Animations panel reveals timing drift.
- **Real devices** for gestures (drawers, swipe).
- **Fresh eyes next day** — imperfections invisible during development surface later.
