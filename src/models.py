from typing import List
from pydantic import BaseModel, Field, AliasChoices
import uuid

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

class UnansweredQuestion(BaseModel):
    # This AliasChoices grabs the real ID from the file if it's named "id"
    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), 
        validation_alias=AliasChoices('question_id', 'id')
    )
    question: str

class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str

class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]

class MinimalSearchResults(BaseModel):
    question_id: str
    question_str: str  # Corrected for Moulinette
    retrieved_sources: List[MinimalSource]

class MinimalAnswer(MinimalSearchResults):
    answer: str

class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int

class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: List[MinimalAnswer]
