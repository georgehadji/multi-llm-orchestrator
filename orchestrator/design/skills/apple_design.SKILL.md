---
name: apple-design
description: Apple's approach to interface design and fluid, physical motion, translated for the web. Use when building or reviewing gesture-driven UI, spring animations, drag/swipe/sheet interactions, momentum and interruptible transitions, translucent materials, and reduced-motion.
---

# Apple Design

How Apple builds interfaces that stop feeling like a computer and start feeling like an extension of you. This knowledge comes from Apple's WWDC design talks — chiefly *Designing Fluid Interfaces* (WWDC 2018) — distilled and translated into the web platform (CSS, Pointer Events, `requestAnimationFrame`, spring libraries like Motion/Framer Motion).

The through-line: **an interface feels alive when motion starts from the current on-screen value, inherits the user's velocity, projects momentum forward, and can be grabbed and reversed at any instant.** Springs are the tool that makes all of this natural, because they are inherently interruptible and velocity-aware.

## The Core Idea

> "When we align the interface to the way we think and move, something magical happens — it stops feeling like a computer and starts feeling like a seamless extension of us."

An interface is fluid when it behaves like the physical world: things respond instantly, move continuously, carry momentum, resist at boundaries, and can be redirected mid-motion.

## 1. Response — kill latency

The moment lag appears, the feeling of directness "falls off a cliff."

- **Respond on pointer-down, not on release.** Highlight a button the instant it's pressed.
- **Feedback must be continuous *during* the interaction, not just at the end.** For a drag, slider, or drawer, update the UI 1:1 with the pointer the whole way through.

## 2. Direct manipulation — 1:1 tracking

When the user drags something, it must stay glued to the finger — and respect the offset from *where they grabbed it*.

- Use Pointer Events with `setPointerCapture` so tracking continues even when the pointer leaves the element's bounds.

## 3. Interruptibility — the single most important principle

Every animation must be interruptible and redirectable at any moment.

- **Never lock out input during a transition.**
- **Always animate from the *presentation* (current) value, never the target value.** On interrupt, read the element's live on-screen transform and start the new animation from there.
- **Avoid CSS transitions and `@keyframes` for anything gesture-driven** — they can't be smoothly grabbed and reversed mid-flight. Springs animate from the current value by default.
- **Decompose 2D motion into independent X and Y springs.**

## 4. Behavior over animation — use springs

A pre-scripted, fixed-duration animation can't respond to new input. A spring can.

Apple deliberately replaced the physics triplet (mass/stiffness/damping) with two designer-friendly parameters:
- **Damping ratio** — controls overshoot. `1.0` = critically damped, no bounce. `< 1.0` = overshoots and oscillates.
- **Response** — how quickly the value reaches the target, in seconds. Lower = snappier.

**Defaults:** Start most UI at damping `1.0`. Add bounce (damping ~`0.8`) only when the gesture itself carried momentum.

## 5. Velocity handoff — the seam between drag and animation

When a gesture ends, the animation must continue at the finger's exact velocity. Pass the pointer's release velocity as the spring's initial velocity.

## 6. Momentum projection — animate to where the gesture is *going*

Use velocity to **project the resting position** — exactly like scroll deceleration:

```js
function project(initialVelocity, decelerationRate = 0.998) {
  return (initialVelocity / 1000) * decelerationRate / (1 - decelerationRate);
}
```

## 7. Spatial consistency — symmetric paths, anchored origins

- **Enter and exit along the same path.** A panel that slides in from the right must dismiss to the right.
- **Anchor interactions to their source.** Set `transform-origin` to the trigger.

## 8. Rubber-banding — soft boundaries

At an edge, resist progressively instead of stopping hard:

```js
function rubberband(overshoot, dimension, constant = 0.55) {
  return (overshoot * dimension * constant) / (dimension + constant * Math.abs(overshoot));
}
```

## 9. Frame-level smoothness

- Animate only compositor-friendly properties — `transform` and `opacity` — and hint with `will-change` where motion is imminent.
- Keep the per-frame positional change below the perception threshold.

## 10. Reduced motion & accessibility

```css
@media (prefers-reduced-motion: reduce) {
  .sheet { transition: opacity 200ms ease; transform: none !important; }
}
@media (prefers-reduced-transparency: reduce) {
  .toolbar { background: white; backdrop-filter: none; }
}
```

## 11. Typography — optical sizing, tracking, leading

- **Tracking (letter-spacing) is size-specific.** Large display text wants negative tracking; small text wants slightly positive tracking.
- **Leading (line-height) tracks size inversely.** Tight on large headings, looser on body copy.

## 12. Design foundations — eight principles

1. **Purpose** — Make with intention; decide what *not* to build.
2. **Agency** — Keep people in control: offer choices, don't force a single path.
3. **Responsibility** — Act in the user's interest. Privacy, safety, anticipate misuse.
4. **Familiarity** — Build on what people already know.
5. **Flexibility** — Design for different contexts, devices, and abilities.
6. **Simplicity** — Strip the unnecessary so the core purpose shines.
7. **Craft** — Uncompromising attention to detail builds trust.
8. **Delight** — The result of getting the other seven right, not confetti tacked on top.

## Quick Reference

| Need | Technique | Concrete value |
| --- | --- | --- |
| Default UI spring | Critically damped, no overshoot | `damping 1.0`, `response 0.3–0.4` |
| Momentum / flick spring | Under-damped, slight bounce | `damping ~0.8`, `response 0.3–0.4` |
| Gesture → spring velocity | Hand off release velocity | normalized: `gestureVelocity / (target − current)` |
| Flick landing point | Project momentum | `current + (v/1000)·d/(1−d)`, `d ≈ 0.998` |
| Interrupt cleanly | Start from presentation (live) value | read the on-screen transform |
| Reversible transition | Mirror the easing curve | inverse cubic-bézier |
| 1:1 drag | Pointer Events + capture | respect the grab offset |
| Boundary | Rubber-band, don't hard-stop | progressive resistance |
| Reduced motion | Cross-fade, not slide/spring | `@media (prefers-reduced-motion)` |
