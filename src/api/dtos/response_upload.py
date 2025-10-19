from typing import List
from pydantic import BaseModel

from src.schema import File


class UploadFilesResponseDto(BaseModel):
    task_id: str
    files: List[File]