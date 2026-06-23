from typing import TypedDict, Literal
from pydantic import BaseModel


class SubTask(BaseModel):
    id: str
    description: str
    assigned_to: Literal["research", "document", "synthesis"]
    status: Literal["pending", "in_progress", "completed", "rejected"]
    output: str = ""
    confidence: float = 0.0


class ExecutionPlan(BaseModel):
    question: str
    subtasks: list[SubTask]


class ReviewResult(BaseModel):
    subtask_id: str
    approved: bool
    confidence: float
    issues: list[str] = []
    rejection_count: int = 0


class GraphState(TypedDict):
    question: str
    document_text: str
    plan: ExecutionPlan
    completed_outputs: dict
    review_history: list[ReviewResult]
    needs_human_review: bool
    human_approved: bool
    final_answer: str
    status: str
