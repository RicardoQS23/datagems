from pydantic import BaseModel, Field
from typing import List, Optional

class QuizGenerationRequest(BaseModel):
    MCQs: List[str] = Field(..., description="List of MCQ CSV URLs")
    numQuizzes: int = Field(10000, gt=0, description="Number of quizzes to generate")
    numMCQs: int = Field(10, gt=0, description="Number of MCQs per quiz")
    listTopics: Optional[List[str]] = Field(default_factory=list, description="List of topics to filter (optional)")
    numTopics: int = Field(10, gt=0, description="Number of topics to sample from the dataset")
    topicMode: bool = Field(1, ge=0, le=1, description="0=same, 1=different")
    levelMode: bool = Field(1, ge=0, le=1, description="0=same, 1=different")
    orderLevel: Optional[int] = Field(2, ge=0, le=2, description="0=ascending, 1=descending by difficulty (optional)")