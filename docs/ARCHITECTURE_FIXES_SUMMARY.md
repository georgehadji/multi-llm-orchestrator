# Architecture Rules Engine - Bug Fixes

**Date:** 2026-03-04

---

## 🔧 Fixes Applied

### 1. Fixed Project Type Detection Priority

**Problem:** Projects mentioning "frontend" were detected as web_frontend even if they explicitly used backend frameworks like FastAPI.

**Example:**
- Input: "FastAPI backend API with HTML frontend"
- Before: Detected as `web_frontend` → TypeScript/React
- After: Detected as `web_api` → Python/FastAPI

**Fix:** Updated `_detect_project_type()` to prioritize backend framework keywords:
```python
# Priority 1: Explicit backend frameworks
backend_frameworks = ["fastapi", "django", "flask", "tornado", "sanic", "starlette"]
if any(word in text for word in backend_frameworks):
    return "web_api"
```

---

### 2. Fixed JSON Parsing for LLM Responses

**Problem:** LLM responses sometimes included markdown code fences (```json) or extra text, causing JSON parse errors.

**Error:**
```
Failed to parse LLM architecture response: Expecting value: line 1 column 1 (char 0)
```

**Fix:** Added robust JSON parsing in both `_generate_rules_with_llm()` and `_optimize_rules_with_llm()`:
```python
# Remove markdown code fences if present
if response_text.startswith("```"):
    lines = response_text.split("\n")
    # ... remove fences

# Try to find JSON in the response
json_match = re.search(r'\{[\s\S]*\}', response_text)
if json_match:
    response_text = json_match.group(0)
```

---

### 3. Fixed Empty Response Handling

**Problem:** Empty LLM responses caused crashes.

**Fix:** Added empty response check:
```python
if not response_text:
    logger.error("LLM returned empty response for architecture decision")
    raise ValueError("Empty LLM response")
```

---

### 4. Fixed generate_summary() Parameter

**Problem:** The method signature still had `model_used` parameter but the implementation now uses metadata fields.

**Fix:** 
- Removed `model_used` parameter from `generate_summary()`
- Updated all call sites in `engine.py` and `architecture_rules.py`

---

## 📊 Test Case: portfolio_python_developer.yaml

### Before Fix:
```
🏗️ ARCHITECTURE DECISION
============================================================
Decided by: LLM (GPT-4o)              ← Wrong format

Style: Event Driven                    ← Wrong (should be Layered/Hexagonal)
Paradigm: Object Oriented
API: GRAPHQL                           ← Wrong (should be REST)
Database: Document

Technology Stack:
  Primary: typescript                  ← WRONG! Should be Python
  Frameworks: react, next.js           ← WRONG! Should be FastAPI
```

### After Fix:
```
🏗️ ARCHITECTURE DECISION
============================================================
Decided by: LLM (Rule-based → Optimized)  ← Correct format

Style: Layered                          ← Correct
Paradigm: Object Oriented
API: REST                               ← Correct
Database: Relational

Technology Stack:
  Primary: python                       ← Correct
  Frameworks: fastapi, pydantic         ← Correct
```

---

## 🎯 Files Modified

| File | Changes |
|------|---------|
| `orchestrator/architecture_rules.py` | Fixed project detection, JSON parsing, empty response handling, summary method |
| `orchestrator/engine.py` | Updated generate_summary() call |

---

## ✅ Verification

To verify the fixes work:

```bash
python -m orchestrator --file projects/portfolio_python_developer.yaml
```

Expected output:
- Primary language: **python** (not typescript)
- Frameworks: **fastapi** (not react/next.js)
- API: **REST** (not GraphQL)
- Decision label: **LLM (Rule-based → Optimized)** or **Rule-based**
