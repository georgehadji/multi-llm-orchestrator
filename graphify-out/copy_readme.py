import shutil
src = r"C:\Users\tesse\.graphify\repos\nousresearch\hermes-agent\README.md"
dst = r"E:\Documents\Vibe-Coding\Ai Orchestrator\graphify-out\hermes_README.md"
shutil.copy2(src, dst)
print("Copied OK")
