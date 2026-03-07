# Εφαρμοσμένες Αλλαγές Routing & Μοντέλων

> ⚠️ **UPDATE 2026-03-03:** Τα GLM μοντέλα (Zhipu) έχουν αφαιρεθεί από το σύστημα.

**Ημερομηνία:** 2025-02-26  
**Status:** ✅ Ολοκληρώθηκε (ιστορικό)

---

## 📋 Αλλαγές που Εφαρμόστηκαν

### 1. Νέα Μοντέλα (`orchestrator/models.py`)

```python
class Model(str, Enum):
    # ➕ ΝΕΟ: OpenAI o4-mini — cost-effective reasoning
    O4_MINI = "o4-mini"
    
    # ➕ ΝΕΟ: Gemini Flash-Lite — cheapest Gemini
    GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"
    
    # ➕ ΝΕΟ: GLM-4-Flash — FREE tier
    GLM_4_FLASH = "glm-4-flash-250414"
```

**Σύνολο μοντέλων:** 9 → 12

---

### 2. Ενημερωμένο Pricing (`COST_TABLE`)

| Μοντέλο | Input | Output | Σημείωση |
|---------|-------|--------|----------|
| **O4_MINI** | $1.50 | $6.00 | Νέο — OpenAI reasoning |
| **GEMINI_FLASH_LITE** | $0.075 | $0.30 | Νέο — Cheapest Gemini |
| **GLM_4_FLASH** | $0.00 | $0.00 | Νέο — **FREE** |

---

### 3. Ενημερωμένα Routing Tables

#### CODE_GEN
```python
# ΠΡΙΝ:
[DEEPSEEK_CODER, MINIMAX_TEXT_01, KIMI_K2_5, GLM_4_PLUS, GPT_4O]

# ΜΕΤΑ:
[DEEPSEEK_CODER, GPT_4O, GPT_4O_MINI, GEMINI_FLASH, KIMI_K2_5]
```
**Αλλαγές:**
- ❌ Αφαίρεση MINIMAX_TEXT_01
- ❌ Αφαίρεση GLM_4_PLUS
- ➕ Προσθήκη GPT_4O_MINI (budget option)

---

#### CODE_REVIEW
```python
# ΠΡΙΝ:
[DEEPSEEK_CODER, MINIMAX_TEXT_01, KIMI_K2_5, GPT_4O]

# ΜΕΤΑ:
[DEEPSEEK_CODER, GPT_4O, GPT_4O_MINI, GEMINI_FLASH]
```
**Αλλαγές:**
- ❌ Αφαίρεση MINIMAX_TEXT_01
- ❌ Αφαίρεση KIMI_K2_5
- ➕ Προσθήκη GPT_4O_MINI
- ➕ Προσθήκη GEMINI_FLASH

---

#### REASONING
```python
# ΠΡΙΝ:
[DEEPSEEK_REASONER, MINIMAX_TEXT_01, KIMI_K2_5, GPT_4O]

# ΜΕΤΑ:
[DEEPSEEK_REASONER, GPT_4O, O4_MINI, GEMINI_PRO, MINIMAX_TEXT_01]
```
**Αλλαγές:**
- ❌ Αφαίρεση KIMI_K2_5
- ➕ Προσθήκη O4_MINI (νέο reasoning)
- ➕ Προσθήκη GEMINI_PRO

---

#### WRITING
```python
# ΠΡΙΝ:
[GLM_4_PLUS, GPT_4O, DEEPSEEK_CODER, GEMINI_PRO]

# ΜΕΤΑ:
[GPT_4O, GEMINI_PRO, DEEPSEEK_CODER, GLM_4_FLASH]
```
**Αλλαγές:**
- ❌ Αφαίρεση GLM_4_PLUS
- ➕ Προσθήκη GLM_4_FLASH (**FREE**)
- ⬆️ GPT_4O → #1

---

#### DATA_EXTRACT
```python
# ΠΡΙΝ:
[GEMINI_FLASH, GPT_4O_MINI, DEEPSEEK_CODER]

# ΜΕΤΑ:
[GEMINI_FLASH_LITE, GPT_4O_MINI, GEMINI_FLASH, GLM_4_FLASH, DEEPSEEK_CODER]
```
**Αλλαγές:**
- ➕ Προσθήκη GEMINI_FLASH_LITE (cheapest)
- ➕ Προσθήκη GLM_4_FLASH (**FREE**)

---

#### SUMMARIZE
```python
# ΠΡΙΝ:
[GEMINI_FLASH, DEEPSEEK_CODER, GPT_4O_MINI]

# ΜΕΤΑ:
[GEMINI_FLASH_LITE, GEMINI_FLASH, GPT_4O_MINI, GLM_4_FLASH]
```
**Αλλαγές:**
- ➕ Προσθήκη GEMINI_FLASH_LITE (cheapest)
- ➕ Προσθήκη GLM_4_FLASH (**FREE**)
- ❌ Αφαίρεση DEEPSEEK_CODER (overkill για summarize)

---

#### EVALUATE
```python
# ΠΡΙΝ:
[DEEPSEEK_CODER, MINIMAX_TEXT_01, KIMI_K2_5, GPT_4O]

# ΜΕΤΑ:
[GPT_4O, DEEPSEEK_CODER, O4_MINI, MINIMAX_TEXT_01]
```
**Αλλαγές:**
- ⬆️ GPT_4O → #1 (πιο αξιόπιστο)
- ❌ Αφαίρεση KIMI_K2_5
- ➕ Προσθήκη O4_MINI

---

### 4. Ενημερωμένα Fallback Chains

```python
# ΝΕΕΣ ΕΓΓΡΑΦΕΣ:
Model.O4_MINI:           Model.DEEPSEEK_REASONER  # Reasoning → Reasoning
Model.GEMINI_FLASH_LITE: Model.GLM_4_FLASH        # Cheap → FREE
Model.GLM_4_FLASH:       Model.GEMINI_FLASH_LITE  # FREE → Cheap

# ΕΝΗΜΕΡΩΜΕΝΕΣ:
Model.DEEPSEEK_REASONER: Model.O4_MINI            # R1 → o4-mini (both reasoning)
Model.GLM_4_PLUS:        Model.GEMINI_PRO         # GLM+ → Gemini Pro
```

---

## 📊 Αντίκτυπος

### Εξοικονόμηση Κόστους (Εκτίμηση)

| Task Type | Πριν (avg) | Μετά (avg) | Εξοικονόμηση |
|-----------|------------|------------|--------------|
| DATA_EXTRACT | $0.20 | $0.12 | **40%** |
| SUMMARIZE | $0.18 | $0.10 | **44%** |
| CODE_GEN | $0.45 | $0.35 | **22%** |
| CODE_REVIEW | $0.42 | $0.32 | **24%** |
| WRITING | $1.10 | $0.95 | **14%** |
| REASONING | $0.85 | $0.75 | **12%** |
| EVALUATE | $0.55 | $0.50 | **9%** |

**Συνολική Εξοικονόμηση: ~25-30%**

---

### Διαθεσιμότητα

- ✅ **12 μοντέλα** (από 9)
- ✅ **Cross-provider** fallbacks για resilience
- ✅ **FREE tier** επιλογή (GLM-4-Flash)

---

## 🔧 Files Modified

```
orchestrator/
├── models.py      # Model enum, COST_TABLE, ROUTING_TABLE, FALLBACK_CHAIN
└── __init__.py    # Export routing tables for inspection
```

---

## ✅ Verification

```python
from orchestrator import Model, COST_TABLE, ROUTING_TABLE

# Έλεγχος νέων μοντέλων
print(Model.GEMINI_FLASH_LITE)  # gemini-2.5-flash-lite
print(Model.GLM_4_FLASH)        # glm-4-flash-250414
print(Model.O4_MINI)            # o4-mini

# Έλεγχος τιμολόγησης
print(COST_TABLE[Model.GLM_4_FLASH])  # {'input': 0.0, 'output': 0.0} — FREE!

# Έλεγχος routing
from orchestrator import TaskType
print(ROUTING_TABLE[TaskType.DATA_EXTRACT][0])  # GEMINI_FLASH_LITE
```

---

## 🎯 Χρήση FREE Tier

```python
from orchestrator import Orchestrator, Budget

# Χρήση GLM-4-Flash (FREE) για budget-conscious tasks
async with Orchestrator(budget=Budget(max_usd=1.0)) as orch:
    # DATA_EXTRACT και SUMMARIZE θα χρησιμοποιήσουν GLM-4-Flash
    # ως fallback όταν το budget είναι περιορισμένο
    state = await orch.run_project(...)
```

---

*Αλλαγές ολοκληρώθηκαν με επιτυχία!*
