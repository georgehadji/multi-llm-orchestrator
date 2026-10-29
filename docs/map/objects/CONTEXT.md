# objects/ — the nouns

**Job:** one card per noun, grouped by how an editor asks, not by folder layout.

## Inputs
[_index.md](_index.md) lists every noun and its status. Template: `../_templates/object.md`. Rules: `../_meta/schema.md`.

## Output
`objects/<cluster>/<Name>.md`, one folder per cluster: `work`, `routing`, `money`, `ports`, `pipeline`. No cluster folder exists until its first card does.

## Rules
- Cite `path:line`. `status: verified` needs a date, a commit and citations.
- "If you change this" is first-order only. "Does not hit" names the obvious next noun that is the wrong one.
- Do not paste audit reports or as-built behaviour into a card. Link the source.

## Human check
Read the card against source before marking it verified.
