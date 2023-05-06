from typing import List

from pydantic import BaseModel


class QueryResponse(BaseModel):
    """Схема ответа"""
    query: str
    vector: List[float]