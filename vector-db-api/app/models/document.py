from pydantic import BaseModel
from typing import Optional, Dict


# Modify to use a vector database
class DocumentBase(BaseModel):
    content: str


class DocumentCreate(DocumentBase):
    pass


class DocumentUpdate(BaseModel):
    content: Optional[str] = None


class Document(DocumentBase):
    id: str


class SearchQuery(BaseModel):
    query: str
    limit: int = 5
