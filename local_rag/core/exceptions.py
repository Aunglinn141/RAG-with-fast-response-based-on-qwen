"""Errors shared between document processing and the HTTP layer."""


class DocumentError(Exception):
    def __init__(self, code: str, message: str, status: int = 422, *, document_id: str | None = None):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status
        self.document_id = document_id
