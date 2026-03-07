# Directory Comparison Instructions

Due to shell execution limitations, please run the comparison script manually:

## Steps to Run Comparison:

1. Open Command Prompt or PowerShell
2. Navigate to the project directory:
   ```
   cd "D:\Vibe-Coding\Ai Orchestrator"
   ```

3. Run the comparison script:
   ```
   python compare_directories.py
   ```

4. The script will generate:
   - `directory_comparison_report.txt` - Human-readable report
   - `directory_comparison.json` - Detailed comparison data

## Alternative: Manual Comparison via PowerShell

```powershell
# Compare orchestrator folder
$ref = Get-ChildItem -Path "D:\Vibe-Coding\Ai Orchestrator\orchestrator" -Recurse -File | Select-Object FullName, Length, LastWriteTime
diff $ref (Get-ChildItem -Path "E:\Documents\Vibe-Coding\Ai Orchestrator\orchestrator" -Recurse -File | Select-Object FullName, Length, LastWriteTime)
```
