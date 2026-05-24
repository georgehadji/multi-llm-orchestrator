# ElevenLabs Design System — Quick Reference

**For AI Prompting & Manual Implementation**

---

## Quick Color Reference

```
Background: Pure White (#ffffff) or Light Gray (#f5f5f5)
Text: Black (#000000)
Secondary text: Dark Gray (#4e4e4e)
Muted text: Warm Gray (#777169)
Warm surface: Warm Stone (rgba(245, 242, 239, 0.8))
Border: #e5e5e5 or rgba(0,0,0,0.05)
```

---

## Typography Quick Reference

| Element | Font | Size | Weight | Line-Height | Letter-Spacing |
|---------|------|------|--------|-------------|----------------|
| Hero Headline | Waldenburg | 48px | 300 | 1.08 | -0.96px |
| Section Heading | Waldenburg | 36px | 300 | 1.17 | normal |
| Card Title | Waldenburg | 32px | 300 | 1.13 | normal |
| Body Large | Inter | 20px | 400 | 1.35 | normal |
| Body | Inter | 18px | 400 | 1.60 | +0.18px |
| UI Text | Inter | 16px | 400 | 1.50 | +0.16px |
| Button | Inter | 15px | 500 | 1.47 | normal |
| CTA Uppercase | WaldenburgFH | 14px | 700 | 1.10 | +0.7px |
| Caption | Inter | 14px | 400 | 1.50 | +0.14px |
| Code | Geist Mono | 13px | 400 | 1.85 | normal |

---

## Example Component Prompts

### Hero Section
```
Create a hero on white background. Headline at 48px Waldenburg weight 300, 
line-height 1.08, letter-spacing -0.96px, black text. Subtitle at 18px Inter 
weight 400, line-height 1.60, letter-spacing 0.18px, #4e4e4e text. Two pill 
buttons: black (9999px radius, 0px 14px padding) and warm stone 
(rgba(245,242,239,0.8), 30px radius, 12px 20px padding, warm shadow 
rgba(78,50,23,0.04) 0px 6px 16px).
```

### Card Design
```
Design a card: white background, 20px radius. Shadow: rgba(0,0,0,0.06) 0px 0px 
0px 1px, rgba(0,0,0,0.04) 0px 1px 2px, rgba(0,0,0,0.04) 0px 2px 4px. Title at 
32px Waldenburg weight 300, body at 16px Inter weight 400 letter-spacing 
0.16px, #4e4e4e.
```

### White Pill Button
```
Create a white pill button: white bg, 9999px radius. Shadow: rgba(0,0,0,0.4) 
0px 0px 1px, rgba(0,0,0,0.04) 0px 4px 4px. Text at 15px Inter weight 500.
```

### Uppercase CTA Label
```
Create an uppercase CTA label: 14px WaldenburgFH weight 700, text-transform 
uppercase, letter-spacing 0.7px.
```

### Navigation Bar
```
Design navigation: white sticky header. Inter 15px weight 500. Black pill CTA 
right-aligned. Border-bottom: rgba(0,0,0,0.05).
```

### Code Block
```
Design a code block: Geist Mono 13px, line-height 1.85. Background: #f5f5f5. 
Padding: 20px. Border-radius: 16px. Text color: #4e4e4e.
```

---

## Shadow Stack Reference

### Level 0 (Flat)
```css
/* No shadow - page background, text blocks */
box-shadow: none;
```

### Level 0.5 (Inset Edge)
```css
/* Internal border definition */
box-shadow: rgba(0,0,0,0.075) 0px 0px 0px 0.5px inset, 
            #fff 0px 0px 0px 0px inset;
```

### Level 1 (Outline Ring)
```css
/* Shadow-as-border for cards */
box-shadow: rgba(0,0,0,0.06) 0px 0px 0px 1px,
            rgba(0,0,0,0.04) 0px 1px 2px,
            rgba(0,0,0,0.04) 0px 2px 4px;
```

### Level 2 (Card)
```css
/* Button elevation, prominent cards */
box-shadow: rgba(0,0,0,0.4) 0px 0px 1px,
            rgba(0,0,0,0.04) 0px 4px 4px;
```

### Level 3 (Warm Lift)
```css
/* Featured CTAs — warm-tinted */
box-shadow: rgba(78,50,23,0.04) 0px 6px 16px;
```

---

## Border Radius Scale

| Use | Radius |
|-----|--------|
| Small links, inline | 2px |
| Nav items, tabs, tags | 4px |
| Small containers | 8px |
| Medium cards, dropdowns | 10-12px |
| Standard cards | 16px |
| Featured cards, code panels | 18-20px |
| Section containers | 24px |
| Warm stone CTA | 30px |
| Primary buttons, nav pills | 9999px |

---

## Spacing System

Base unit: 8px

```
1px, 3px, 4px, 8px, 9px, 10px, 11px, 12px, 16px, 18px, 20px, 24px, 28px, 32px, 40px
```

---

## Key Characteristics Checklist

- [ ] Near-white canvas with warm undertones (#f5f5f5, #f5f2ef)
- [ ] Waldenburg weight 300 (light) for display — ethereal, whisper-thin headings
- [ ] Inter with positive letter-spacing (0.14-0.18px) for body — airy readability
- [ ] Multi-layered shadow stacks at sub-0.1 opacity — surfaces barely exist
- [ ] Pill buttons (9999px) with warm stone-tinted backgrounds
- [ ] WaldenburgFH bold uppercase for specific CTA button labels only
- [ ] Warm shadow tints: rgba(78, 50, 23, 0.04) — shadows have color
- [ ] Geist Mono / ui-monospace for code snippets
- [ ] Apple-like generous whitespace — massive vertical spacing
- [ ] Warm emptiness — whitespace has tactile, physical quality

---

## Common Mistakes to Avoid

❌ Using bold (700) Waldenburg for headings  
✅ Use weight 300 only — the lightness IS the brand

❌ Heavy shadows (>0.1 opacity)  
✅ Keep all shadows at whisper-level depth (<0.1)

❌ Cool gray borders  
✅ Use warm-tinted shadows and borders throughout

❌ Skip inset shadow component  
✅ Include half-pixel inset borders (0.5px inset)

❌ Negative letter-spacing on body text  
✅ Inter uses positive tracking (+0.14px to +0.18px)

❌ Sharp corners (<8px) on cards  
✅ Generous radius (16px-24px) is structural

❌ Brand colors  
✅ Achromatic with warm undertones only

❌ Opaque, heavy buttons  
✅ Warm translucent stone treatment is the signature

---

## CSS Variables (Copy-Paste)

```css
:root {
  --white: #ffffff;
  --light-gray: #f5f5f5;
  --warm-stone: #f5f2ef;
  --black: #000000;
  --dark-gray: #4e4e4e;
  --warm-gray: #777169;
  --border: #e5e5e5;
  --border-subtle: rgba(0, 0, 0, 0.05);
  
  --shadow-inset: rgba(0, 0, 0, 0.075) 0px 0px 0px 0.5px inset;
  --shadow-outline: rgba(0, 0, 0, 0.06) 0px 0px 0px 1px;
  --shadow-soft: rgba(0, 0, 0, 0.04) 0px 4px 4px;
  --shadow-card: rgba(0, 0, 0, 0.4) 0px 0px 1px, rgba(0, 0, 0, 0.04) 0px 4px 4px;
  --shadow-warm: rgba(78, 50, 23, 0.04) 0px 6px 16px;
  
  --font-display: 'Waldenburg', sans-serif;
  --font-display-bold: 'WaldenburgFH', sans-serif;
  --font-body: 'Inter', sans-serif;
  --font-mono: 'Geist Mono', monospace;
  
  --radius-pill: 9999px;
  --radius-button-warm: 30px;
  --radius-card: 16px;
  --radius-large: 20px;
}
```

---

**Use this guide for prompting AI design tools or manual implementation.**
