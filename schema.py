# schema.py
from typing import List, Literal
from pydantic import BaseModel, Field

class TechnicalDecision(BaseModel):
    id: str = Field(description="Unique ID for the decision (e.g., dec-001)")
    title: str = Field(description="Short title of the technical decision")
    summary: str = Field(description="Concise summary of why and what was decided")
    tags: List[str] = Field(description="Technical tags like 'db', 'auth', 'frontend'")
    observed_at: str = Field(description="ISO timestamp when this was recorded")

class UIRule(BaseModel):
    id: str = Field(description="Unique ID (e.g., rule-001)")
    rule: str = Field(description="The actual UI/UX rule or guideline")
    scope: Literal["ui", "ux", "localization"] = Field(description="The scope of the rule")
    notes: str = Field(description="Additional context or exceptions")

class CriticalWarning(BaseModel):
    id: str = Field(description="Unique ID (e.g., warn-001)")
    area: str = Field(description="The system area affected (e.g., security, deployment)")
    message: str = Field(description="The warning message")
    severity: Literal["low", "medium", "high"] = Field(description="Severity level")

class ProjectKnowledgeBase(BaseModel):
    """The root schema for all extracted project data"""
    decisions: List[TechnicalDecision]
    rules: List[UIRule]
    warnings: List[CriticalWarning]