# 🚀 LLM Orchestrator Dashboard v6.5

## ✨ Concurrent Projects + Connected Status

### 🆕 v6.5 - Concurrent Projects!

✅ **Δείχνει "Connected"** όταν το server είναι online  
✅ **Πολλαπλά projects** μπορούν να τρέχουν ταυτόχρονα  
✅ **New + Improve + Upload** όλα μαζί!

### Προηγούμενα Features
- ✅ Auto API Connection on startup
- ✅ Real-time API Status Panel
- ✅ Live Console & Progress

## 🎮 Εκκίνηση

```bash
# Windows (double-click)
Start_Mission_Control.bat

# Ή από command line
python run_mission_control_standalone.py
```

## 🔌 API Connections Panel

Στην κορυφή του dashboard θα δεις:

| Status | Εικονίδιο | Σημασία |
|--------|-----------|---------|
| 🟢 Connected | ✓ Πράσινο | API είναι online και έτοιμο |
| 🟡 No API Key | ⚠ Κίτρινο | Λείπει το API key (πρέπει να το προσθέσεις στο .env) |
| 🔴 Error | ✗ Κόκκινο | Σφάλμα σύνδεσης |

## 📋 Πώς λειτουργεί

1. **Ξεκινάς το dashboard**
2. **Αυτόματα ελέγχει** όλα τα APIs:
   - OpenAI (GPT-4o, GPT-4o-mini)
   - DeepSeek (Coder, Reasoner)
   - Google (Gemini)
   - Kimi (K2.5)
   - Minimax

3. **Εμφανίζει status** για κάθε provider
4. **Κουμπί Reconnect** για να ξανα-ελέγξεις

## 🔧 Προσθήκη API Keys

Δημιούργησε ένα `.env` αρχείο στο root:

```env
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
KIMI_API_KEY=sk-...
MINIMAX_API_KEY=...
```

## 🎯 Features

- ✅ **Auto-connect** on startup
- ✅ **Real-time status** για κάθε API
- ✅ **Reconnect button** για manual refresh
- ✅ **New Project** - Δημιουργία νέου project
- ✅ **Improve Codebase** - Βελτίωση υπάρχοντος κώδικα
- ✅ **Upload Project** - Upload και επεξεργασία αρχείων
- ✅ **Live Console** - Real-time logs
- ✅ **Progress Tracking** - Progress bar και task monitoring

## 🌐 URL

Μετ την εκκίνηση, άνοιξε:
**http://localhost:8888**

## 🚀 Ready to use!
