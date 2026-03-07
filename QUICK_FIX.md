# 🚨 Quick Fix

## Πρόβλημα: "Disconnected | Starting..."

Το dashboard δείχνει ότι είναι disconnected. Αυτό σημαίνει ότι το WebSocket δεν συνδέεται.

## Λύση 1: Hard Refresh

1. **Κλείσε** το browser tab
2. **Άνοιξε** νέο tab
3. Πήγαινε σε `http://localhost:8888`
4. Πάτα **Ctrl+F5** (hard refresh)

## Λύση 2: Επανεκκίνηση Server

1. **Κλείσε** το terminal (Ctrl+C ή X)
2. Περίμενε 5 δευτερόλεπτα
3. Τρέξε: `Start_Mission_Control.bat`

## Λύση 3: Διαφορετικό Browser

Δοκίμασε άλλο browser (Chrome, Firefox, Edge).

## Λύση 4: Έλεγχος με curl

Αν έχεις curl installed:
```bash
curl http://localhost:8888/api/state
```

Αν δεις JSON output, το server δουλεύει.
Αν δεις "connection refused", το server δεν τρέχει.

## Συχνό Πρόβλημα: Projects από προηγούμενη εκτέλεση

Αν το dashboard θυμάται παλιά projects:
1. Κλείσε το server
2. Διέγραψε το φάκελο `uploads/` (αν υπάρχει)
3. Ξεκίνα ξανά
