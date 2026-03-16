# 自己从零开始编写的 BPE 分词器
# BPE (Byte Pair Encoding) 字节对编码
import os
import sys
import io
import nbformat
import types

# 从 Jupyter Notebook 中导入指定函数或类
def import_from_notebook():

    # 内部函数：读取 notebook 文件并导入指定的函数或类
    def import_definitions_from_notebook(fullname, names):

        # 获取当前工作目录
        current_dir = os.getcwd()

        # 构造 notebook 文件路径
        # 这里会到上级目录 ../05_bpe-from-scratch/ 查找 notebook 文件
        path = os.path.join(current_dir, "..", "05_bpe-from-scratch", fullname + ".ipynb")

        # 标准化路径（去掉 .. 等）
        path = os.path.normpath(path)

        # 加载 notebook 文件
        if not os.path.exists(path):
            raise FileNotFoundError(f"Notebook file not found at: {path}")

        # 读取 notebook 内容
        with io.open(path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # 创建一个新的 Python 模块对象
        # 用来存放从 notebook 中导入的类和函数
        mod = types.ModuleType(fullname)

        # 把模块注册到 sys.modules 中
        # 这样 Python 认为这个模块已经被导入
        sys.modules[fullname] = mod

        # 遍历 notebook 中的每个 cell
        for cell in nb.cells:

            # 只处理代码单元
            if cell.cell_type == "code":

                cell_code = cell.source

                # 遍历需要导入的函数/类名
                for name in names:

                    # 判断 cell 中是否定义了指定函数或类
                    if f"def {name}" in cell_code or f"class {name}" in cell_code:

                        # 执行该代码单元
                        # 执行环境是 mod.__dict__
                        # 这样函数或类就会被加载到模块中
                        exec(cell_code, mod.__dict__)

        # 返回包含导入内容的模块
        return mod

    # notebook 文件名（不带 .ipynb）
    fullname = "bpe-from-scratch"

    # 需要导入的类或函数名
    names = ["BPETokenizerSimple"]

    # 调用内部函数执行导入
    return import_definitions_from_notebook(fullname, names)


# 从 notebook 中导入模块
imported_module = import_from_notebook()

# 从模块中获取 BPETokenizerSimple 类
BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)

# 创建 tokenizer 实例
tokenizer_gpt2 = BPETokenizerSimple()

# 从 OpenAI GPT-2 的词表文件加载 tokenizer 数据
tokenizer_gpt2.load_vocab_and_merges_from_openai(

    # GPT2 vocabulary 文件
    vocab_path=os.path.join("gpt2_model", "encoder.json"),

    # BPE merge 规则文件
    bpe_merges_path=os.path.join("gpt2_model", "vocab.bpe")
)

# 待编码文本
text = "Hello, world. Is this-- a test?"

# 使用 tokenizer 将文本转换为 token id
integers = tokenizer_gpt2.encode(text)

# 打印 token id 结果
print(integers)