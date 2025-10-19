class ApiException(Exception):
    def __init__(self, message: str, description: str, code: int):
        self.message = message
        self.description = description
        self.code = code