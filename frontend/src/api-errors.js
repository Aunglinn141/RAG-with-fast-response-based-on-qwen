export class ApiError extends Error {
  constructor(message, code = 'request_failed', documentId = null, details = []) {
    super(message);
    this.code = code;
    this.documentId = documentId;
    this.details = details;
  }
}
