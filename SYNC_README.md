# 🔄 AI Orchestrator - Sync & Cleanup Guide

## Γρήγορη Εκκίνηση

### 1️⃣ Preview (Δες τι θα γίνει)

```bash
cd "D:\Vibe-Coding\Ai Orchestrator"
python full_sync_and_cleanup.py --dry-run
```

### 2️⃣ Execute (Τρέξε το sync & cleanup)

```bash
python full_sync_and_cleanup.py --apply
```

---

## 📋 Τι κάνει το script

### ✅ Συγχρονισμός (D: → E:)
- Μεταφέρει όλα τα `.py` files από `orchestrator/`
- Μεταφέρει όλα τα `.md` files από `docs/`
- Μεταφέρει test files
- Μεταφέρει config files (pyproject.toml, README.md, κλπ)
- **Συγκρίνει** timestamps και μεταφέρει μόνο αν χρειάζεται

### 🗑️ Cleanup (Και στα 2 drives)
Διαγράφει προσωρινά αρχεία:
- Backup files (`.bak`)
- Temp files (`temp_*.py`)
- Old session logs
- Reorganization plans (προσωρινά docs)

### 🔍 Verification
Ελέγχει ότι τα critical files υπάρχουν και στα 2 locations:
- `issue_tracking.py`
- `slack_integration.py`
- `git_service.py`
- `git_hooks.py`
- `docs/ISSUE_TRACKING.md`

---

## 📁 Δημιουργήθηκαν αυτά τα αρχεία

| Αρχείο | Τοποθεσία | Περιγραφή |
|--------|-----------|-----------|
| `full_sync_and_cleanup.py` | D: + E: | Main sync script |
| `sync_d_to_e.py` | D: | Simple sync only |
| `sync_from_d.py` | E: | Simple sync (alt) |
| `SYNC_D_TO_E.bat` | D: | Windows batch |
| `SYNC_README.md` | D: | This guide |

---

## 🎯 Εκτέλεση βήμα-βήμα

### Βήμα 1: Preview
```bash
python full_sync_and_cleanup.py --dry-run
```
Θα δεις:
```
[14:32:01] STEP 1: Syncing Files (D: → E:)
[14:32:01] 📦 Core Python modules...
[14:32:02]   ...copied 10 files
[14:32:03]   ✓ Core Python modules: 88 copied
[14:32:03] 📄 Root config files...
[14:32:03]   ✓ pyproject.toml
...
[14:32:05] ⚠️  This was a DRY RUN. No changes were made.
```

### Βήμα 2: Execute
```bash
python full_sync_and_cleanup.py --apply
```
Θα δεις:
```
[14:33:01] ✅ Sync and cleanup complete!

Next steps:
  1. Verify files in E: drive
  2. Run tests to ensure everything works
  3. Commit changes to git
```

---

## 🔍 Επαλήθευση μετά το sync

### Έλεγξε ότι υπάρχουν τα νέα files

```bash
# Στο E: drive
cd "E:\Documents\Vibe-Coding\Ai Orchestrator"
dir orchestrator\issue_tracking.py
dir orchestrator\slack_integration.py
dir orchestrator\git_service.py
dir docs\ISSUE_TRACKING.md
```

### Έλεγξε τα exports

```python
python -c "from orchestrator import IssueTrackerService, SlackNotifier; print('✓ OK')"
```

---

## 📊 Τι θα αλλάξει

### Πριν το sync:
```
D: (88 files)  ======≠======  E: (? files)
     ↑                        
  SOURCE                    TARGET
  (latest)                  (outdated)
```

### Μετά το sync:
```
D: (88 files)  ==============  E: (88 files)
     ↑                          
  SOURCE                      TARGET
  (latest)                    (synced)
```

---

## ⚠️ Προσοχή

- **D: είναι το source of truth** - Πάντα sync D: → E:, όχι το αντίστροφο
- **Backup** - Αν έχεις local αλλαγές στο E:, κράτα backup πρώτα
- **Git** - Το sync δεν επηρεάζει τα git repositories (άλλα folders)

---

## 🚀 Μετά το sync

1. ✅ Επαλήθευσε ότι τα files υπάρχουν
2. 🧪 Τρέξε tests: `python -m pytest tests/`
3. 📦 Commit στο git (αν χρειάζεται)
4. 🗑️ Τρέξε το root folder reorganization και στα 2 drives

---

## ❓ Troubleshooting

### "D: drive not found"
- Βεβαιώσου ότι το `D:	Vibe-Coding	t Orchestrator` υπάρχει
- Τρέξε από το D: drive

### "Permission denied"
- Τρέξε το Command Prompt ως Administrator
- Ή χρησιμοποίησε το `SYNC_D_TO_E.bat` (κάνει right-click → Run as admin)

### Κάποια files λείπουν
- Τρέξε το verification step: `python full_sync_and_cleanup.py --dry-run`
- Έλεγξε το `SYNC_STATUS_REPORT.md` για λίστα critical files
