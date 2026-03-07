# Quick Start: Nash Stability Features
## Ξεκίνα σε 5 λεπτά

Αυτός ο οδηγός θα σου δείξει πώς να ξεκινήσεις με τα νέα Nash stability features.

---

## 📋 Προαπαιτούμενα

```bash
# Έχεις ήδη εγκατεστημένο το orchestrator
pip install -e .

# Ή ενημέρωσε το υπάρχον
pip install --upgrade multi-llm-orchestrator
```

---

## 🚀 Επιλογή 1: Γρήγορο Ξεκίνημα (2 λεπτά)

### Βήμα 1: Δες το τρέχον status
```bash
python -m orchestrator nash-status
```

**Αναμενόμενο output:**
```
============================================================
                   NASH STABILITY REPORT                    
============================================================

  Stability Score: 0.00 [░░░░░░░░░░░░░░░░░░░░] 
  Status: Early stage - minimal switching costs accumulated

  💰 Switching Cost: $0.00
     • Local Value: $0.00
     • Global Value: $0.00

  📊 Accumulated Assets:
     • Knowledge Graph: 0 relationships
     • Learned Patterns: 0
     ...

============================================================
```

> **Σημείωση:** Το σκορ είναι 0 γιατί μόλις ξεκίνησες. Θα αυξηθεί με τη χρήση.

---

## 🚀 Επιλογή 2: Πλήρες Ξεκίνημα (5 λεπτά)

### Βήμα 1: Δημιούργησε ένα Nash-Stable Orchestrator

```python
# start_nash.py
import asyncio
from orchestrator import NashStableOrchestrator, Budget

async def main():
    # Δημιούργησε το orchestrator με όλα τα features
    orchestrator = NashStableOrchestrator(
        budget=Budget(max_usd=5.0),
        org_id="my-company",           # Το όνομά σου
        privacy_budget=1.0,            # Για federated learning
        enable_federation=True,        # Συμμετοχή στο global network
    )
    
    print("✅ Nash-Stable Orchestrator initialized!")
    
    # Τρέξε ένα project
    result = await orchestrator.run_project(
        project_description="Build a FastAPI REST API with JWT authentication",
        success_criteria="All endpoints tested, OpenAPI docs complete",
        budget=3.0,
        enable_learning=True,          # Μάθε από αυτό το run
    )
    
    print(f"\n📊 Project completed: {result['status']}")
    print(f"💰 Cost: ${result.get('cost', 0):.2f}")
    
    # Δες το stability report
    report = orchestrator.get_nash_stability_report()
    print(f"\n🛡️  Nash Stability Score: {report['nash_stability_score']:.2f}")
    print(f"💵 Switching Cost: ${report['switching_cost_analysis']['total_switching_cost_usd']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
```

```bash
python start_nash.py
```

---

### Βήμα 2: Κάνε backup αμέσως

```bash
# Μετά το πρώτο run, κάνε backup
python -m orchestrator nash-backup
```

**Αναμενόμενο output:**
```
Creating backup ...
✓ Backup created: nash_backup_20260303_143022
  Components: 5
  Total size: 12.5 KB
  Estimated value: $15.30
  Checksum: a3f7d9e2b4c8f1a5
```

---

### Βήμα 3: Παρακολούθησε το status

```bash
# Σε άλλο terminal, τρέξε:
python -m orchestrator nash-status --watch
```

Αυτό θα ενημερώνεται κάθε 5 δευτερόλεπτα καθώς τρέχουν projects.

---

## 🎯 Daily Workflow

### 1. Πρωί - Έλεγξε το Status
```bash
python -m orchestrator nash-status
```

### 2. Κατά τη διάρκεια της ημέρας - Τρέξε Projects
```python
# Τα projects αυτόματα ενημερώνουν το knowledge graph
result = await orchestrator.run_project(
    "Build a React component",
    "Component renders without errors",
    budget=2.0,
)
```

### 3. Βράδυ - Κάνε Backup
```bash
python -m orchestrator nash-backup
```

---

## 📊 Παρακολούθηση Progress

### Week 1: Early Stage
```bash
$ python -m orchestrator nash-status
Stability Score: 0.15 [███░░░░░░░░░░░░░░░░░]
Status: Growing - some knowledge accumulated
Switching Cost: $45.50
```

### Week 4: Moderate
```bash
$ python -m orchestrator nash-status
Stability Score: 0.52 [██████████░░░░░░░░░░]
Status: Moderate - meaningful switching costs exist
Switching Cost: $187.30
```

### Month 3: Strong
```bash
$ python -m orchestrator nash-status
Stability Score: 0.78 [███████████████░░░░░]
Status: Strong - significant competitive moat
Switching Cost: $523.80
```

---

## 🛠️ Χρήσιμα Commands

### CLI Commands Cheat Sheet

```bash
# 🔍 Status & Monitoring
python -m orchestrator nash-status                    # Τρέχον status
python -m orchestrator nash-status --watch            # Real-time monitoring
python -m orchestrator nash-status --format json      # Machine readable

# 💾 Backup Management
python -m orchestrator nash-backup                    # Create backup
python -m orchestrator nash-backup --list            # List all backups
python -m orchestrator nash-backup --value           # Show estimated value
python -m orchestrator nash-backup --restore <file>  # Restore from backup

# ⚙️ Auto-Tuning
python -m orchestrator nash-tuning                    # Show tuning status
python -m orchestrator nash-tuning --tune <param> --value <val>  # Manual tune
python -m orchestrator nash-tuning --drift-check     # Check for drift

# 📊 Model Comparison
python -m orchestrator nash-compare deepseek-chat gpt-4o
python -m orchestrator nash-compare deepseek-chat gpt-4o --task-type CODE_GEN

# 📡 Event Monitoring
python -m orchestrator nash-events                    # Recent events
python -m orchestrator nash-events --follow          # Real-time events
python -m orchestrator nash-events --type DRIFT_DETECTED
```

---

## 🔧 Python API Examples

### Example 1: Custom Event Handler
```python
from orchestrator import get_event_bus, EventType

# Κάνε subscribe σε events
@get_event_bus().on(EventType.TEMPLATE_CONVERGED)
async def on_convergence(event):
    print(f"🎉 Template {event.data['variant_name']} converged!")
    print(f"EMA Score: {event.data['ema_score']:.2f}")

# Αυτό θα τρέξει αυτόματα όταν ένα template συγκλίνει
```

### Example 2: Compare Models Programmatically
```python
from orchestrator import get_cost_quality_frontier, Model, TaskType

frontier = get_cost_quality_frontier()

comparison = frontier.compare_models(
    Model.DEEPSEEK_CHAT,
    Model.GPT_4O,
    TaskType.CODE_GEN,
)

print(f"Quality difference: {comparison['differences']['quality']:+.3f}")
print(f"Recommendation: {comparison['recommendation']}")
```

### Example 3: Manual Backup
```python
from orchestrator import get_backup_manager

async def backup_and_upload():
    mgr = get_backup_manager()
    
    # Create backup
    manifest = await mgr.create_backup()
    print(f"Backup created: {manifest.backup_id}")
    print(f"Value: ${manifest.estimated_value_usd:.2f}")
    
    # Upload to cloud (δικό σου implementation)
    # await upload_to_s3(f"{manifest.backup_id}.tar.gz")

# Τρέξε κάθε μέρα
import schedule
import time

schedule.every().day.at("02:00").do(lambda: asyncio.run(backup_and_upload()))

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Example 4: Auto-Tuning Integration
```python
from orchestrator import get_auto_tuner

tuner = get_auto_tuner()

# Μετά από κάθε task
async def after_task(task_result):
    # Ενημέρωσε το auto-tuner
    result = await tuner.tune(
        "template_exploration_rate",
        metric_value=task_result.quality_score,
    )
    
    if result:
        print(f"Auto-tuned: {result.old_value:.2f} → {result.new_value:.2f}")
        print(f"Reason: {result.reason}")
```

---

## 🚨 Troubleshooting

### Πρόβλημα: "No module named 'orchestrator.nash_events'"
```bash
# Λύση: Ενημέρωσε το installation
pip install -e .
# Ή
python -m pip install --upgrade -e .
```

### Πρόβλημα: Stability Score παραμένει 0
```bash
# Λύση: Βεβαιώσου ότι enable_learning=True
# Και ότι τρέχεις τουλάχιστον 3-4 projects

python -c "
from orchestrator import NashStableOrchestrator
orch = NashStableOrchestrator()
print(f'Components initialized:')
print(f'  KG: {orch.knowledge_graph is not None}')
print(f'  Templates: {orch.adaptive_templates is not None}')
print(f'  Frontier: {orch.pareto_frontier is not None}')
print(f'  Federated: {orch.federated is not None}')
"
```

### Πρόβλημα: Backup αποτυγχάνει
```bash
# Έλεγξε permissions
ls -la .nash_backups/

# Ή δημιούργησε manual το directory
mkdir -p .nash_backups
chmod 755 .nash_backups
```

---

## 🎯 Next Steps

Αφού ξεκινήσεις:

1. **Τρέξε 5-10 projects** για να accumulάρεις αρχική γνώση
2. **Κάνε backup** μετά από κάθε σημαντικό milestone
3. **Παρακολούθησε το stability score** - στόχος είναι > 0.7
4. **Συμμετοχή στο federated learning** - ενεργοποίησε το `enable_federation=True`
5. **Ρύθμισε scheduled backups** - daily ή weekly

---

## 📈 Αναμενόμενη Εξέλιξη

| Timeline | Stability Score | Switching Cost | Αντίκτυπος |
|----------|-----------------|----------------|------------|
| Day 1 | 0.00 | $0 | Ξεκίνημα |
| Week 1 | 0.15 | $45 | Early stage |
| Month 1 | 0.45 | $180 | Moderate |
| Month 3 | 0.78 | $520 | Strong moat |
| Month 6 | 0.88 | $950+ | Nash stable |

---

## 💡 Pro Tips

1. **Μην διαγράφεις τα `.nash_*` directories** - περιέχουν το accumulated knowledge σου
2. **Κάνε backup πριν από major updates** - προστασία από data loss
3. **Χρησιμοποίησε το `--watch` mode** όταν κάνεις debug
4. **Σύγκρινε models τακτικά** - οι τιμές αλλάζουν με το χρόνο
5. **Ενεργοποίησε το federation** - βοηθάει το global network και εσένα

---

## 🆘 Χρειάζεσαι Βοήθεια;

```bash
# Γενική βοήθεια
python -m orchestrator --help

# Βοήθεια για specific command
python -m orchestrator nash-status --help
python -m orchestrator nash-backup --help
python -m orchestrator nash-tuning --help
```

---

**Ξεκίνα τώρα:** Δοκίμασε το `python -m orchestrator nash-status` και δες το status σου! 🚀
