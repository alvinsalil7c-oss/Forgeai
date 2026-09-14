"""
Stage 1 verification script.

Run this after creating the virtual environment and installing the
REQUIRED NOW packages (opencv-python, mediapipe, numpy, pandas).

It only checks that the libraries import correctly and prints their
versions. It does NOT touch the webcam or microphone yet - that
starts in Stage 2.
"""

import sys

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}\n")

# (module import name, pip package name)
checks = [
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
]

all_ok = True

for module_name, pip_name in checks:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK]   {pip_name:<15} version {version}")
    except ImportError as e:
        all_ok = False
        print(f"[FAIL] {pip_name:<15} not installed correctly -> {e}")

print()
if all_ok:
    print("Stage 1 environment looks good. Ready for Stage 2 (webcam test).")
else:
    print("Fix the [FAIL] lines above (re-run the matching pip install command) before continuing.")
    sys.exit(1)