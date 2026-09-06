# Component generation for generated websites

How the orchestrator steers generated websites toward real component libraries,
and how to target Svelte as well as React.

Added 2026-09-06. Status of each claim below is stated explicitly — see
**Verification status** at the end for what could not be checked in this
environment.

## What this is

`orchestrator/design/component_registry.py` curates *references* to components
from upstream libraries — not code. Each `ComponentSpec` carries a name, its
source library, layout/animation/accessibility attributes, and a
`prompt_reference` describing the component in prose. `get_component_prompt()`
composes that with the project's `DesignSystem` tokens into the prompt handed to
the model.

That design is why the orchestrator can cite shadcn, Aceternity, bits-ui and
friends without vendoring them, adding npm dependencies, or running a bundler:
the model writes the component, the registry tells it *what to write*.

## It was completely dead until now

The registry imported `ComponentSource` and `ComponentSpec` from
`design_system`, and **neither type was defined anywhere in the repository**
(hunt T17 confirmed `ComponentSource` was referenced 17 times, defined zero
times). So `from .design_system import ComponentSource` raised `ImportError`,
and `generators/website_generator.py` silently fell through to a
`_FakeRegistry` returning four bare section names:

```python
async def select_components(self, **kw):
    return [_FakeComponent(n) for n in ["hero", "features", "pricing", "contact"]]
```

Every generated site therefore lost all component curation and all design-token
guidance. Defining the two missing types revives 524 lines and 17 curated
components across 7 sections.

## Targeting a framework

```bash
python -m orchestrator website -d "a landing page for a bakery" --framework svelte
python -m orchestrator website -d "a SaaS pricing page"        --framework react
python -m orchestrator website -d "a brochure site"            --framework html
```

`--framework` accepts `html`, `react`, `next.js`, `svelte`, `sveltekit`.
`next.js` collapses onto React's component ecosystem and `sveltekit` onto
Svelte's, via `normalize_framework()`.

The framework decides three things:

| | react / next.js | svelte / sveltekit | html |
|---|---|---|---|
| Component sources offered | shadcn, Aceternity, Magic UI, 21st, reactbits, animate-ui | bits-ui, sveltebits | none (framework-agnostic only) |
| File extension | `.tsx` | `.svelte` | `.html` |
| OUTPUT contract in the prompt | React/Next.js + TypeScript types | Svelte 5 single-file, runes, `<script lang="ts">` | semantic HTML5 + inlined `<style>` |

A Svelte build is never steered toward a React-only library. When a section has
no framework-native component curated (for example, there is no Svelte-native
pricing component yet), selection falls back to the best framework-agnostic
candidate and logs that it did — returning *something* matters more than
purity, because the page still needs that section.

## Adding a component or a library

Add a source to `ComponentSource` in `orchestrator/design_system.py` and give it
a framework in `COMPONENT_SOURCE_FRAMEWORKS`; everything downstream — selection,
prompting, extension — follows from that one mapping. Then add `ComponentSpec`
entries to `COMPONENT_LIBRARY` in `design/component_registry.py`, keyed by
section.

Write `prompt_reference` as a description of behaviour and layout, not as code.
It is prose handed to a model, so it should say what the component *is* and how
it should adapt on mobile, not how to implement it.

## Scope: what this does not do

- **No SvelteKit project scaffolding.** React/Next.js targets get
  `_assemble_nextjs_page()` plus `_verify_and_fix_build()` (npm install and a
  real build). There is no SvelteKit equivalent, so a `--framework svelte` run
  produces correct `.svelte` components but not an installable, build-verified
  SvelteKit project. That scaffolder is a separate, substantial piece of work;
  a half-built one that emits a project which does not build would be worse than
  none.
- **No npm dependency management** for the component libraries. Nothing is
  installed or pinned; the model is steered by description.
- **No MCP integration.** The four libraries publish MCP servers
  (bits-ui, sveltebits, reactbits, animate-ui). Wiring those would let the
  registry pull real, current component catalogues instead of curated prose.
  The seam for that is `ComponentRegistry._get_candidates()`.

## Verification status

Honest accounting of what was and was not checked:

- **Verified by test** (`tests/unit/test_component_generation_frameworks.py`,
  27 tests; full suite 2611 passed, +27, zero regressions): the registry imports and yields real `ComponentSpec`s; the live
  generator resolves to it rather than `_FakeRegistry`; framework normalization;
  React and Svelte source sets are disjoint apart from `CUSTOM`; a Svelte build
  selects bits-ui/sveltebits; a React build is unchanged; sections without a
  native match still return a component; the OUTPUT contract and file extension
  per framework; `--framework svelte` parses; scoring bounds; and that a plain
  string `tone` no longer raises.
- **NOT verified — the framework attribution of the four newly added
  libraries.** This environment's egress proxy blocks `bits-ui.com`,
  `sveltebits.xyz`, `reactbits.dev` and `animate-ui.com`, and their MCP servers
  are not connected here, so their docs could not be read. The assignments in
  `COMPONENT_SOURCE_FRAMEWORKS` follow each project's name and widely-known
  framework. Three are self-evident from their names; **animate-ui is the one
  worth double-checking**. If any is wrong, correct that one mapping and
  selection follows automatically.
- **NOT verified — generated output quality.** No real generation run was made
  (that spends budget against live providers). The tests prove the *prompt and
  selection* are correct, not that a model produces good Svelte from them.
