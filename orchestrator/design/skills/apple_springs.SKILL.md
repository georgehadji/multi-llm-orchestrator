---
name: apple-springs
description: Quick-reference extracted values from Apple's WWDC fluid-interface guidelines — spring configs, damping/response pairs, momentum projection, rubber-banding, and gesture feel checklist. Use when tuning spring parameters or implementing gesture-driven motion.
---

# Apple Springs & Fluid Motion — Quick Reference

Extracted from Apple's *Designing Fluid Interfaces* (WWDC 2018). These are the exact concrete values Apple ships. Cite these in code reviews and generation tasks.

## Spring Parameters (Apple's Two-Parameter System)

Apple uses **damping ratio** + **response** instead of mass/stiffness/damping:

- **Damping ratio** — controls overshoot. `1.0` = critically damped (no bounce). `< 1.0` = under-damped (bounces). Lower = bouncier.
- **Response** — how quickly the value reaches the target, in seconds. Lower = snappier. This is NOT "duration" — a spring has no fixed duration; its settle time emerges from the parameters.

### Apple-Ship Values

| Interaction | Damping | Response |
| --- | --- | --- |
| Move / reposition (e.g. PiP) | `1.0` | `0.4` |
| Rotation | `0.8` | `0.4` |
| Drawer / sheet | `0.8` | `0.3` |

### Web Mapping (Motion / Framer Motion)

```js
// Critically damped default (no overshoot)
animate(el, { y: 0 }, { type: 'spring', bounce: 0, duration: 0.4 });

// Momentum interaction — a little bounce, only because a flick preceded it
animate(el, { y: target }, { type: 'spring', bounce: 0.2, duration: 0.4 });
```

A safe house style: `damping: 1.0` springs everywhere by default. Reserve bounce for momentum-driven, physical interactions.

## Velocity Handoff

Pass the pointer's release velocity as the spring's initial velocity. Some spring APIs want **relative** velocity:

```
relativeVelocity = gestureVelocity / (targetValue − currentValue)
```

Framer Motion / Motion take absolute px/s velocity directly (`velocity` option), so you usually hand it the raw value.

## Momentum Projection

Apple's exact projection function (from the *Designing Fluid Interfaces* sample code):

```js
// decelerationRate ≈ 0.998 for normal scroll feel; 0.99 for snappier
function project(initialVelocity, decelerationRate = 0.998) {
  return (initialVelocity / 1000) * decelerationRate / (1 - decelerationRate);
}

const projectedEndpoint = currentPosition + project(releaseVelocity);
const target = nearestSnapPoint(projectedEndpoint);
animateSpringTo(target, { velocity: releaseVelocity });
```

Note: the physics-textbook `v²/(2·decel)` is *not* what Apple ships — use the exponential-decay form above.

## Rubber-Banding (Soft Boundaries)

At an edge, resist progressively instead of stopping hard:

```js
function rubberband(overshoot, dimension, constant = 0.55) {
  return (overshoot * dimension * constant) / (dimension + constant * Math.abs(overshoot));
}
```

## Gesture Feel Checklist

- **Tap:** highlight on touch-*down* (instant), commit on touch-*up*. Add ~10px of hysteresis/hit padding.
- **Drag/swipe:** require a small movement threshold (hysteresis, ~10px) before committing to a direction, then track 1:1.
- **Detect all plausible gestures in parallel from the first move**, then confidently cancel the losers once intent is clear.
- **Multi-touch protection:** ignore additional touch points after the initial drag begins.
- **Minimize disambiguation delays.** Double-tap detection unavoidably delays single taps; only pay that cost where double-tap truly exists.

## Translucent Materials

```css
.toolbar {
  background: rgba(255, 255, 255, 0.6);
  backdrop-filter: blur(20px) saturate(180%);
  border-top: 1px solid rgba(255, 255, 255, 0.4);
}
```

- **Material weight encodes hierarchy:** darker/heavier = structural, lighter = interactive.
- **Dim to focus, separate to keep flow.** A modal task pairs with a dimming scrim. A parallel panel uses translucency without a scrim.
- **Materialize, don't just fade.** Animate blur radius and scale together on enter/exit.

## Reduced Motion

```css
@media (prefers-reduced-motion: reduce) {
  .sheet { transition: opacity 200ms ease; transform: none !important; }
}
@media (prefers-reduced-transparency: reduce) {
  .toolbar { background: white; backdrop-filter: none; }
}
```

Reduced motion doesn't mean *no* feedback — keep opacity/color changes that aid comprehension.
