from importlib.metadata import version
import sys
print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

print("python version:", sys.version)