import uuid
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone # Use timezone-aware UTC
from typing import Literal, Dict, Any, List, Optional, Union

# --- Nested Models for Specific Signal Types ---

class ReasoningSignal(BaseModel):
    """Analysis of reasoning patterns (CoT, ToT)."""
    pattern_type: Optional[Literal["CoT", "ToT"]] = Field(None, description="Detected reasoning pattern type (Chain-of-Thought or Tree-of-Thought).")
    steps_detected: int = Field(0, description="Number of distinct steps detected (e.g., numbered list items).")
    branches_detected: int = Field(0, description="Number of distinct options or branches detected.")
    trigger_phrase: Optional[str] = Field(None, description="Specific text phrase that triggered the pattern detection.")

class ToolSignal(BaseModel):
    """Analysis of tool usage patterns (code, functions, commands)."""
    code_execution_detected: bool = Field(False, description="Whether code blocks (e.g., ```python ... ```) were found.")
    code_languages: List[str] = Field(default_factory=list, description="List of programming languages identified in code blocks.")
    function_tool_calls: List[str] = Field(default_factory=list, description="List of function/tool names identified in potential JSON call structures.")
    command_line_detected: bool = Field(False, description="Whether command-line execution patterns (e.g., pip, curl) were detected.")

class RetrievalSignal(BaseModel):
    """Analysis of retrieval augmentation patterns."""
    retrieval_detected: bool = Field(False, description="Whether patterns indicating retrieval from external knowledge were found.")
    retrieval_source_count: int = Field(0, description="Number of distinct citations (URLs, file names, references) detected.")
    retrieval_citations: List[str] = Field(default_factory=list, description="List of detected citations or source references.")

class SelfReferenceSignal(BaseModel):
    """Analysis of self-referential statements."""
    self_reference_present: bool = Field(False, description="Whether the LLM referred to itself as an AI, model, etc.")
    self_reference_phrase: Optional[str] = Field(None, description="The specific phrase detected indicating self-reference.")

class ContextUseSignal(BaseModel):
    """Analysis of references to conversation history."""
    context_history_used: bool = Field(False, description="Whether patterns referencing previous turns were detected.")
    context_reference_count: int = Field(0, description="Number of phrases detected referencing conversation history.")

class CognitiveSignals(BaseModel):
    """Structured analysis of cognitive signals detected in LLM responses."""
    reasoning: ReasoningSignal = Field(default_factory=ReasoningSignal)
    tools: ToolSignal = Field(default_factory=ToolSignal)
    retrieval: RetrievalSignal = Field(default_factory=RetrievalSignal)
    self_reference: SelfReferenceSignal = Field(default_factory=SelfReferenceSignal)
    context_use: ContextUseSignal = Field(default_factory=ContextUseSignal) # Renamed from context
    analysis_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(),
                                    description="ISO timestamp of when the analysis was performed.")
    status: Optional[str] = Field(None, description="Status of the analysis (e.g., 'no_text_response' if analysis skipped).")


# --- Main Schema Models ---

class TokenUsage(BaseModel):
    """Token usage information provided by some LLM APIs."""
    prompt_tokens: Optional[int] = Field(None, description="Number of tokens in the input prompt.")
    completion_tokens: Optional[int] = Field(None, description="Number of tokens generated in the response.")
    total_tokens: Optional[int] = Field(None, description="Total number of tokens processed.")

class ResponseMetadata(BaseModel):
    """
    Metadata associated with an LLM response, including performance metrics,
    token counts, and cognitive signal analysis.
    """
    latency_seconds: float = Field(..., description="Response time duration in seconds.") # Renamed, made required
    tokens: Optional[TokenUsage] = Field(None, description="Token usage counts, if available from the source API.")
    raw_response: Optional[Dict[str, Any]] = Field(None, description="Optional: The raw, unprocessed JSON response from the source API.")
    cognitive_signals: Optional[CognitiveSignals] = Field(None, description="Optional: Analysis of detected cognitive signals.")
    context: Optional[Any] = Field(None, description="Optional: Context returned by some APIs (e.g., Ollama).") # Added context

    class Config:
        # Ensure TokenUsage and CognitiveSignals are validated if provided
        validate_assignment = True


class ParsedLLMEntry(BaseModel):
    """
    Unified schema representing a parsed LLM interaction entry, normalized
    from potentially different source API formats. This structure is typically
    used before saving to the database.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the interaction.")
    source: Literal["LM Studio", "Ollama", "OpenAI"] # Add more sources if needed
    type: Literal["text", "image", "embed"] = Field(..., description="The primary type of the response content.")
    model: str = Field(..., description="Identifier of the LLM model used.")
    prompt: Union[str, List[Dict[str, str]]] = Field(..., description="The input prompt (string) or messages (list of dicts).")
    # Response can be text, embedding list, raw bytes for image, or None if stream failed etc.
    response: Union[str, List[float], bytes, None] = Field(..., description="The generated response content.")
    # Default factory creates an empty metadata obj; fields like latency should be added after call
    metadata: ResponseMetadata = Field(..., description="Metadata including latency, tokens, signals, etc.")
    # Use timezone-aware UTC timestamp
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc),
                                description="Timestamp when the interaction object was created (UTC).")

    class Config:
        arbitrary_types_allowed = True # Necessary to allow 'bytes' type for images
        validate_assignment = True # Validate fields when they are assigned after initialization

    # Optional validator example (can be useful)
    # @validator('metadata')
    # def check_latency_positive(cls, v):
    #     if v.latency_seconds < 0:
    #         raise ValueError('Latency cannot be negative')
    #     return v