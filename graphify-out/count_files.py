from pathlib import Path
import json
repo = Path(r"C:\Users\tesse\.graphify\repos\nousresearch\hermes-agent")
py_files = list(repo.rglob("*.py"))
md_files = list(repo.rglob("*.md"))
json_files = list(repo.rglob("*.json"))
toml_yaml = list(repo.rglob("*.toml")) + list(repo.rglob("*.yaml")) + list(repo.rglob("*.yml"))
dirs = {}
for f in py_files:
    rel = f.parent.relative_to(repo)
    parts = rel.parts
    root = parts[0] if parts else "."
    dirs[root] = dirs.get(root, 0) + 1
total_py = len(py_files)
total_md = len(md_files)
total_words = sum(sum(1 for _ in open(f, encoding="utf-8", errors="replace")) for f in list(py_files)[:100])
result = {
    "py_files": total_py,
    "md_files": len(md_files),
    "json_files": len(json_files),
    "yaml_toml_files": len(toml_yaml),
    "all_files": total_py + len(md_files) + len(json_files) + len(toml_yaml),
    "dirs_by_py": dict(sorted(dirs.items(), key=lambda x: -x[1])[:30])
}
print(json.dumps(result, indent=2))
