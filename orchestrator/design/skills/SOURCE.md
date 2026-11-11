# taste-skill upstream source

**Repository:** https://github.com/Leonxlnx/taste-skill  
**Last pushed:** 2026-05-26T19:31:39Z  
**License:** MIT — Copyright (c) 2026 Leonxlnx  

## Files vendored

| Local filename | Upstream path |
|---|---|
| `default.SKILL.md` | `skills/taste-skill/SKILL.md` |
| `redesign.SKILL.md` | `skills/redesign-skill/SKILL.md` |
| `soft.SKILL.md` | `skills/soft-skill/SKILL.md` |
| `minimalist.SKILL.md` | `skills/minimalist-skill/SKILL.md` |
| `brutalist.SKILL.md` | `skills/brutalist-skill/SKILL.md` |
| `imagegen_web.SKILL.md` | `skills/imagegen-frontend-web/SKILL.md` |
| `image_to_code.SKILL.md` | `skills/image-to-code-skill/SKILL.md` |
| `animation_vocabulary.SKILL.md` | `skills/animation-vocabulary/SKILL.md` |
| `apple_design.SKILL.md` | `skills/apple-design/SKILL.md` |
| `apple_springs.SKILL.md` | `skills/apple-design/SKILL.md` (extracted reference) |
| `review_animations.SKILL.md` | `skills/review-animations/SKILL.md` |
| `animation_standards.SKILL.md` | `skills/review-animations/STANDARDS.md` |

**Animation skill upstream source:** https://github.com/emilkowalski/skills

## senior-frontend (separate upstream)

| Local filename | Upstream path |
|---|---|
| `senior_frontend.SKILL.md` | `cli-tool/components/skills/development/senior-frontend/SKILL.md` |

**Repository:** https://github.com/davila7/claude-code-templates (listing: https://www.aitmpl.com/component/skills/development/senior-frontend)
**Version vendored:** 2.0.0, fetched 2026-11-11, unmodified.
**License:** MIT, author claude-code-templates. The skill's own workflow text is original to that project.
**Not vendored:** `references/` (ui-checklist.md, react-patterns.md, NOTICE.md: adapted from Vercel Labs and Addy Osmani MIT material) and `scripts/` (a Node + Playwright browser checker). They add about 22 KB per frontend task, and this model only writes code, so it cannot run the checker. `taste_skill_service.py` prefixes the injected text with a note saying so.
**Refresh:** `scripts/refresh_taste_skills.py` does not cover this file; re-fetch it with `gh api -H "Accept: application/vnd.github.raw" repos/davila7/claude-code-templates/contents/<path above>`.

## awwwards-mcp (not vendored, run at build time)

`design/awwwards_client.py` runs the npm package `awwwards-mcp` (https://github.com/INSANE0777/Awwwards-mcp, MIT, Node >= 22.13) through `npx -y awwwards-mcp@<pinned version>` and asks it for a design brief before an LLM website build. The version is pinned in `PACKAGE`; bump it deliberately, and prefer a release at least two weeks old. On by default; `ORCH_AWWWARDS_ENABLED=false` turns it off. The upstream README limits use to personal, non-commercial inspiration, and screenshots and content stay with Awwwards and the site creators.

## To refresh

```bash
python scripts/refresh_taste_skills.py
```

Or manually re-run the gh api commands in Phase 0 of the integration plan.
