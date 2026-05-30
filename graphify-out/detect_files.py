import json
from graphify.detect import detect
from pathlib import Path

INPUT_PATH = Path(r"C:\Users\tesse\.graphify\repos\nousresearch\hermes-agent")
result = detect(INPUT_PATH)
print(json.dumps(result, indent=2))
