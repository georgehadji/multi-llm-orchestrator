# Plateau Model Escalation Feature

**Version:** v6.1 | **Date:** 2026-03-04

---

## 🎯 Overview

Όταν ένα task φτάνει σε **plateau** (δεν βελτιώνεται άλλο) αλλά έχει **χαμηλό score**, το σύστημα πλέον δοκιμάζει αυτόματα ένα **ανώτερο μοντέλο LLM** αντί να σταματάει.

---

## 🤔 Πρόβλημα

### Πριν
```
task_009 iter 1: score=0.45
task_009 iter 2: score=0.47  
task_009 iter 3: score=0.46
task_009: plateau at low score after 3 iters (Δ=0.0100, best=0.460)
→ Status: DEGRADED (υποβαθμισμένο)
```

Το task απέτυχε γιατί το μοντέλο "κόλλησε" σε χαμηλό score.

---

## ✅ Λύση: Model Escalation

### Μετά
```
task_009 iter 1: score=0.45
task_009 iter 2: score=0.47
task_009 iter 3: score=0.46
task_009: plateau at low score (best=0.460), escalating from deepseek-chat to gpt-4o

task_009 iter 4: score=0.78  ✨ Βελτίωση!
task_009 iter 5: score=0.85  ✓ Threshold met!
→ Status: COMPLETED
```

---

## 🔧 How It Works

### Trigger Conditions

Η escalation γίνεται όταν **όλες** οι συνθήκες είναι αληθείς:

| Condition | Value | Description |
|-----------|-------|-------------|
| Plateau detected | `Δ < 0.02` | Καμία βελτίωση μεταξύ iterations |
| Low score | `score < threshold × 0.6` | Αρκετά χαμηλό για να χρειάζεται βοήθεια |
| Min score | `score ≥ threshold × 0.3` | Όχι τόσο χαμηλό που είναι άσκοπο |
| Budget available | `can_afford(0.05)` | Υπάρχει budget για το επόμενο μοντέλο |
| Not escalated yet | `model_escalated = False` | Μόνο μία escalation per task |

### Model Tiers

```
CHEAP (Tier 0)          BALANCED (Tier 1)       PREMIUM (Tier 2)
─────────────────────────────────────────────────────────────────
Gemini Flash Lite       Gemini Flash            GPT-4o
GPT-4o Mini             DeepSeek Chat           DeepSeek Reasoner
                        Kimi K2.5               Gemini Pro
```

**Escalation flow:**
- Tier 0 → Tier 1
- Tier 1 → Tier 2
- Tier 2 → None (ήδη στο καλύτερο)

### Warm Start

Το νέο μοντέλο λαμβάνει:
1. **Original prompt** — Το αρχικό task description
2. **Previous attempt** — Το best_output από το προηγούμενο μοντέλο
3. **Context** — Το score του προηγούμενου μοντέλου

```python
full_prompt = (
    f"{task.prompt}\n\n"
    f"--- PREVIOUS ATTEMPT (score: {best_score:.2f}) ---\n"
    f"This is a previous attempt that needs improvement:\n"
    f"{best_output}\n\n"
    f"--- YOUR TASK ---\n"
    f"Improve upon the previous attempt to achieve a higher quality score."
)
```

---

## 📊 Παραδείγματα

### Example 1: Επιτυχής Escalation

```python
# Task: Generate complex algorithm
threshold = 0.85

# Iteration με DeepSeek Chat
Iter 1: score=0.42
Iter 2: score=0.45 (Δ=0.03)
Iter 3: score=0.44 (Δ=0.01) ← Plateau detected

# Conditions check:
# - Δ=0.01 < 0.02 ✓
# - score=0.44 < 0.85×0.6=0.51 ✓  
# - score=0.44 ≥ 0.85×0.3=0.255 ✓
# - budget OK ✓
# - not escalated yet ✓

→ Escalating to GPT-4o

# Iteration με GPT-4o  
Iter 4: score=0.76
Iter 5: score=0.88 (≥ 0.85) ✓

→ Status: COMPLETED
```

### Example 2: Ήδη Καλό Score

```python
# Task: Generate API endpoint
threshold = 0.85

Iter 1: score=0.72
Iter 2: score=0.73 (Δ=0.01) ← Plateau detected

# Check: score=0.73 ≥ 0.85×0.5=0.425
# Το score είναι αρκετά καλό, απλά σταματάμε

→ Status: COMPLETED (με score 0.73)
```

### Example 3: Πολύ Χαμηλό Score

```python
# Task: Complex reasoning
threshold = 0.90

Iter 1: score=0.15
Iter 2: score=0.18
Iter 3: score=0.17 (Δ=0.01) ← Plateau detected

# Check: score=0.17 < 0.90×0.3=0.27
# Πολύ χαμηλό, μάλλον λάθος approach

→ Give up (δεν αξίζει escalation)
→ Status: DEGRADED
```

---

## 💰 Cost Considerations

### Budget Protection

```python
# Ελέγχεται πριν το escalation
if not self.budget.can_afford(0.05):
    logger.warning("Budget insufficient for model escalation")
    break
```

- **Minimum cost:** $0.05 για escalation attempt
- **Max escalations:** 1 per task (αποφυγή infinite loops)
- **Fallback:** Αν δεν υπάρχει budget, σταματάει κανονικά

### Cost-Benefit

| Scenario | Cost | Benefit |
|----------|------|---------|
| Escalation + Success | +$0.05-0.20 | Task completed vs degraded |
| Escalation + Failure | +$0.05-0.20 | Γνωρίζουμε ότι δεν γίνεται |
| No escalation | $0 | Task marked as degraded |

---

## 🔍 Implementation Details

### Modified Files

| File | Changes |
|------|---------|
| `orchestrator/engine.py` | Plateau handling, model escalation logic |

### New Method

```python
def _get_next_tier_model(self, current_model: Model, task_type: TaskType) -> Optional[Model]
```

Επιλέγει το επόμενο tier μοντέλο από το FALLBACK_CHAIN.

### Modified Loop Variables

```python
model_escalated = False  # Track if we've tried escalation
```

---

## 📈 Benefits

1. **Higher Success Rate** — Λιγότερα degraded tasks
2. **Cost Efficient** — Μόνο όταν χρειάζεται (χαμηλό score + plateau)
3. **Transparent** — Logging δείχνει πότε γίνεται escalation
4. **Smart** — Warm start με προηγούμενο output
5. **Safe** — Budget checks και μόνο 1 escalation per task

---

## 🚀 Usage

Δεν χρειάζεται καμία αλλαγή! Το feature ενεργοποιείται αυτόματα.

```bash
python -m orchestrator --file projects/example.yaml
```

Θα δεις στο log:
```
task_XXX: plateau at low score (best=0.XXX), escalating from MODEL1 to MODEL2
```

---

## 📝 Configuration (Future)

Πιθανές μελλοντικές ρυθμίσεις:

```yaml
# .orchestrator-config.yml
plateau_escalation:
  enabled: true
  min_score_threshold: 0.3      # threshold × 0.3
  max_score_threshold: 0.6      # threshold × 0.6
  min_budget_usd: 0.05
  max_escalations_per_task: 1
```

---

**🎉 Το Plateau Model Escalation είναι έτοιμο!**
