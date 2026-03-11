import os
import sys
import io
import nbformat
import types

def import_from_notebook():
    def import_definitions_from_notebook(fullname, names):
        current_dir = os.getcwd()
        path = os.path.join(current_dir, "..", "05_bpe-from-scratch", fullname + ".ipynb")
        path = os.path.normpath(path)

        # Load the notebook
        if not os.path.exists(path):
            raise FileNotFoundError(f"Notebook file not found at: {path}")

        with io.open(path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Create a module to store the imported functions and classes
        mod = types.ModuleType(fullname)
        sys.modules[fullname] = mod

        # Go through the notebook cells and only execute function or class definitions
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell_code = cell.source
                for name in names:
                    # Check for function or class definitions
                    if f"def {name}" in cell_code or f"class {name}" in cell_code:
                        exec(cell_code, mod.__dict__)
        return mod

    fullname = "bpe-from-scratch"
    names = ["BPETokenizerSimple"]

    return import_definitions_from_notebook(fullname, names)


imported_module = import_from_notebook()
BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)

tokenizer_gpt2 = BPETokenizerSimple()
tokenizer_gpt2.load_vocab_and_merges_from_openai(
    vocab_path=os.path.join("gpt2_model", "encoder.json"),
    bpe_merges_path=os.path.join("gpt2_model", "vocab.bpe")
)
text = "Hello, world. Is this-- a test?"
integers = tokenizer_gpt2.encode(text)

print(integers)