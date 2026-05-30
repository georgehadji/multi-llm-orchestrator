import sys
with open('graphify-out/.graphify_python', 'w') as f:
    f.write(sys.executable)
print(f"Python: {sys.executable}")
