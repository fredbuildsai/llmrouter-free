"""llmrouter-free: quota-aware LLM failover router built for free-tier providers."""

from llmrouter_free.config import build_router, load_routes, scale_request_timeout
from llmrouter_free.context_budget import TaskBudget, recommend_num_ctx
from llmrouter_free.router import AllDeploymentsExhausted, Deployment, LLMResult, LLMRouter
from llmrouter_free.validation import json_schema_response_format, json_validator

__version__ = "0.1.0"

__all__ = [
    "AllDeploymentsExhausted",
    "Deployment",
    "LLMResult",
    "LLMRouter",
    "TaskBudget",
    "__version__",
    "build_router",
    "json_schema_response_format",
    "json_validator",
    "load_routes",
    "recommend_num_ctx",
    "scale_request_timeout",
]
