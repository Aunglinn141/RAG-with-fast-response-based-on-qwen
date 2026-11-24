"""Shared inference failures for local model backends."""


class ModelError(RuntimeError):
    """An actionable configuration, connection, or inference error."""


class ModelUnavailableError(ModelError):
    """A model is missing or lacks the required capability."""


class ModelOutputLimitError(ModelError):
    """The model exhausted its generation budget before completing."""

    def __init__(self, num_predict):
        self.num_predict = num_predict
        super().__init__(f"回答达到生成长度上限（{num_predict} token），输出尚未完成。")
