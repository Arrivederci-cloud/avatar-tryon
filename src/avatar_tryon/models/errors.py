"""适配器共用的异常。"""


class ModelNotWired(RuntimeError):
    """适配器已定义但尚未接入具体实现。

    这是刻意的：本仓库不分发第三方模型代码与权重，
    接入方式见 docs/model-selection.md。
    """
