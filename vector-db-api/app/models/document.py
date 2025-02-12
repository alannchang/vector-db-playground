from pydantic import BaseModel
from typing import Optional, Dict


# Modify to use a vector database
class DocumentBase(BaseModel):
    content: str
    metadata: Dict = {}


class DocumentCreate(DocumentBase):
    pass


class DocumentUpdate(BaseModel):
    content: Optional[str] = None
    metadata: Optional[Dict] = None


class Document(DocumentBase):
    id: str


class SearchQuery(BaseModel):
    query: str
    limit: int = 5
