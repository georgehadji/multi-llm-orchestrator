# Temporary fix file - will be used to patch the original

OLD_LINES = """        "rationale": "Explanation of the optimized architecture"
    }}}}
}}

If can_optimize is false, set changes to empty array and optimized_architecture to null."""

NEW_LINES = """        "rationale": "Explanation of the optimized architecture"
    }}}
}}}}

If can_optimize is false, set changes to empty array and optimized_architecture to null."""

# Read the file
with open("orchestrator/architecture_rules.py", encoding="utf-8") as f:
    content = f.read()

# Replace
if OLD_LINES in content:
    content = content.replace(OLD_LINES, NEW_LINES)
    with open("orchestrator/architecture_rules.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Fixed!")
else:
    print("Pattern not found")
    # Show what's actually there
    import re

    match = re.search(r'"rationale".*?null\.', content, re.DOTALL)
    if match:
        print("Found:")
        print(repr(match.group(0)[:200]))
