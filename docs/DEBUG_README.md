# 🔧 Dashboard Debugging

## Γιατί δείχνει "Disconnected | Starting...";

Αυτό σημαίνει ότι:
1. Το WebSocket δεν συνδέεται ("Disconnected")
2. Το server status είναι "Starting..." αντί για "Connected"

## Βήματα για διάγνωση

### Βήμα 1: Έλεγχος συντακτικών λαθών
```bash
python diagnose_dashboard.py
```

Αν δείξει ❌, υπάρχει syntax error στον κώδικα.

### Βήμα 2: Έλεγχος αν τρέχει το server
```bash
python check_server.py
```

Αυτό ελέγχει:
- Αν απαντάει το main page
- Αν απαντάει το /api/state
- Αν δουλεύει το WebSocket

### Βήμα 3: Εκκίνηση με debug output
```bash
python start_dashboard_debug.py
```

Αυτό ξεκινάει το server και δείχνει αναλυτικά errors αν υπάρχουν.

## Συνήθη προβλήματα

### Πρόβλημα: "Module not found"
**Λύση:** Τρέξε από το σωστό directory
```bash
cd "E:\Documents\Vibe-Coding\Ai Orchestrator"
python start_dashboard_debug.py
```

### Πρόβλημα: "Port already in use"
**Λύση:** Κλείσε το παλιό terminal και ξαναπροσπάθησε.

### Πρόβλημα: JavaScript errors
**Λύση:** Άνοιξε το browser console (F12 → Console) για να δεις τα errors.

## Επείγουσα επιδιόρθωση

Αν τίποτα δεν δουλεύει, δοκίμασε αυτό:

1. Κλείσε ΟΛΑ τα terminal windows
2. Κάνε refresh τη σελίδα (Ctrl+F5)
3. Τρέξε: `python start_dashboard_debug.py`
4. Περίμενε να δεις "🚀 Mission Control" στο terminal
5. Άνοιξε http://localhost:8888
