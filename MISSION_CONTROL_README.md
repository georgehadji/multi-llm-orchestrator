# 🚀 Mission Control v6.0

## Εκκίνηση

```bash
python start_mission_control.py
```

Ή απευθείας:

```bash
python -c "from orchestrator import run_mission_control; run_mission_control()"
```

## 🌐 URL

Ανοίγει αυτόματα: **http://localhost:8888**

## ✨ Features

### 1. 📝 Project Starter Form
- **Project Type**: Επίλεξε μεταξύ:
  - Full-Stack Application
  - Front-End Only
  - Back-End Only
  - Mobile App
  - WordPress Plugin
  - AI Agent
  - Custom
- **Prompt**: Περιγραφή του project
- **Success Criteria**: Κριτήρια επιτυχίας
- **Budget**: Μέγιστο κόστος σε USD

### 2. 📊 Real-Time Monitoring
Όταν ξεκινάει ένα project βλέπεις:
- ⚡ **Progress Bar**: Συνολική πρόοδος
- 🎯 **Current Task**: Ποιο task τρέχει τώρα
- 📈 **Task Counter**: X από Y tasks ολοκληρώθηκαν
- ⏱️ **Elapsed Time**: Χρόνος εκτέλεσης

### 3. 🏗️ Architecture Panel
Αυτόματη ανάλυση αρχιτεκτονικής:
- **Style**: (π.χ. Single Page Application, API Server)
- **Paradigm**: (π.χ. OOP, Functional)
- **Pattern**: (π.χ. MVC, MVVM)
- **Database**: (π.χ. PostgreSQL, MySQL)
- **Languages**: TypeScript, Python, κλπ
- **Frameworks**: React, FastAPI, κλπ
- **Libraries**: Tailwind, Zustand, κλπ

### 4. 🤖 Models in Use
Βλέπεις σε real-time:
- Ποια LLM χρησιμοποιούνται
- Πόσα calls έγιναν
- Κόστος ανά model

### 5. 📋 Available Models
Προβολή όλων των διαθέσιμων μοντέλων:
- GPT-4o, GPT-4o-mini
- DeepSeek Coder, Reasoner
- Gemini 2.5 Pro
- Kimi K2.5
- Και άλλα...

## 🎨 Interface

### Dark Theme
Το dashboard έχει σκούρο θέμα για άνετη εργασία.

### Live Updates
- WebSocket connection για real-time ενημερώσεις
- Δεν χρειάζεται refresh

### Responsive
Λειτουργεί σε desktop, tablet, και mobile.

## 🔧 Παράδειγμα Χρήσης

1. **Ξεκίνα το dashboard**:
   ```bash
   python start_mission_control.py
   ```

2. **Γράψε ένα prompt**:
   ```
   "Φτιάξε μια TODO εφαρμογή με React και FastAPI"
   ```

3. **Επίλεξε τύπο**:
   ```
   Full-Stack Application
   ```

4. **Πάτα Start Project**:
   - Το dashboard θα δείξει την αρχιτεκτονική
   - Θα ξεκινήσει τα tasks
   - Θα ενημερώνει το progress

5. **Παρακολούθησε**:
   - Πόση πρόοδος έγινε
   - Ποια task τρέχει
   - Ποια models χρησιμοποιούνται

## 🛑 Stop Project

Μπορείς να σταματήσεις το project ανά πάσα στιγμή με το κουμπί "Stop".
