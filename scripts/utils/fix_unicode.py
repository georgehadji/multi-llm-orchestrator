#!/usr/bin/env python3
"""Fix unicode characters in dashboard_mission_control.py"""
import re

filepath = "orchestrator/dashboard_mission_control.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Replace emojis with ASCII equivalents
replacements = {
    "✅": "[OK]",
    "⚠️": "[WARN]",
    "❌": "[ERR]",
    "🚀": "[START]",
    "📁": "[DIR]",
    "📂": "[OUT]",
    "🎯": "[TARGET]",
    "💓": "[BEAT]",
    "🛑": "[STOP]",
    "✨": "[*]",
    "🔴": "[OFF]",
    "🟢": "[ON]",
    "🔧": "[FIX]",
    "📄": "[FILE]",
    "💡": "[IDEA]",
    "📊": "[DATA]",
    "💻": "[CODE]",
    "⚙️": "[CFG]",
    "📡": "[NET]",
    "🌐": "[WEB]",
    "🤖": "[BOT]",
    "🔍": "[FIND]",
    "📝": "[EDIT]",
    "🎉": "[DONE]",
    "🏆": "[WIN]",
    "📈": "[UP]",
    "🔥": "[HOT]",
    "💰": "[$]",
    "⏱️": "[TIME]",
    "⏹️": "[HALT]",
    "🎮": "[GAME]",
    "🐛": "[BUG]",
    "🧹": "[CLEAN]",
    "📋": "[LIST]",
    "🏗️": "[BUILD]",
    "🎨": "[ART]",
    "🔌": "[PLUG]",
    "⚡": "[FAST]",
    "🔄": "[SYNC]",
    "✔️": "[OK]",
    "☑️": "[OK]",
}

for emoji, replacement in replacements.items():
    content = content.replace(emoji, replacement)

# Also fix any remaining emoji using regex
import re
content = re.sub(r'[^\x00-\x7F]+', lambda m: '[?]' if m.group() not in ['\n', '\r', '\t'] else m.group(), content)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed unicode characters")
print("Please restart the server")
