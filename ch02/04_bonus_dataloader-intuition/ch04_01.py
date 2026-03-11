from importlib.metadata import version


print("torch version:", version("torch"))


with open("number-data.txt", "w", encoding="utf-8") as f:
    for number in range(1001):
        f.write(f"{number} ")