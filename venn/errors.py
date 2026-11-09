"""The error type raised for invalid requests; the HTTP layer maps it to a JSON error response."""


class ContractError(Exception):
    """An invalid request. ``code`` is machine-readable, ``message`` is for people, ``status`` is the HTTP status."""

    def __init__(self, code, message, status=422):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status
