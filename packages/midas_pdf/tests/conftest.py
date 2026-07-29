import os

# libomp is double-loaded on macOS (numpy + torch); allow it before torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
