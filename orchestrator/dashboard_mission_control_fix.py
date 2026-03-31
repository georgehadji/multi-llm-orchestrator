#!/usr/bin/env python3
# Fix script to add folder browser

with open("orchestrator/dashboard_mission_control.py", encoding="utf-8") as f:
    lines = f.readlines()

# Find and replace lines 1094-1098 (indices 1093-1097)
new_lines = """                    <div class="mb-4">
                        <label class="block text-sm text-gray-400 mb-2">Codebase Path <span class="text-red-500">*</span></label>
                        <div class="flex gap-2">
                            <input type="text" id="improve-path" class="input-field flex-1" placeholder="C:\\\\Projects\\\\my-app or /home/user/projects/my-app">
                            <button type="button" onclick="browseFolder()" class="bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded text-sm whitespace-nowrap">
                                <i class="fas fa-folder-open mr-2"></i>Browse
                            </button>
                        </div>
                        <p class="text-xs text-gray-500 mt-1">Select project folder using Windows Explorer</p>
                        <input type="file" id="folder-picker" class="hidden" webkitdirectory directory onchange="handleFolderSelect(event)">
                    </div>
"""

# Replace lines 1094-1098 (0-indexed: 1093-1097)
result = lines[:1093] + [new_lines] + lines[1098:]

with open("orchestrator/dashboard_mission_control.py", "w", encoding="utf-8") as f:
    f.writelines(result)

print("✅ HTML updated")
