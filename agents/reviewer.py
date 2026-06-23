import json
import re
import logging

from agents.supervisor import llm_call

logger = logging.getLogger(__name__)

REJECTION_LIMIT = 2
CONFIDENCE_THRESHOLD = 0.4


def review_output(subtask_description: str, output: str, source_context: str,
                   confidence: float, prior_rejections: int = 0) -> dict:
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "approved": False,
            "confidence": confidence,
            "issues": [f"Specialist self-reported confidence {confidence} below threshold {CONFIDENCE_THRESHOLD}"],
            "rejection_count": prior_rejections + 1,
        }

    prompt = f"""You are reviewing an AI agent's output for accuracy.

Task: {subtask_description}

Source material the agent had access to:
{source_context}

Agent's output:
{output}

Does the output contain any claim NOT supported by the source material? Reply ONLY with this JSON:
{{"approved": true/false, "issues": ["list any unsupported claims found"]}}"""

    resp = llm_call(prompt, max_tokens=200)
    try:
        match = re.search(r'\{.*\}', resp, re.DOTALL)
        if match:
            result = json.loads(match.group())
            approved = result.get("approved", True)
            issues = result.get("issues", [])
            return {
                "approved": approved,
                "confidence": confidence,
                "issues": issues,
                "rejection_count": prior_rejections + (0 if approved else 1),
            }
    except Exception as e:
        logger.warning(f"Review parse failed: {e}")

    return {
        "approved": True,
        "confidence": confidence,
        "issues": [],
        "rejection_count": prior_rejections,
    }


def should_escalate_to_human(rejection_count: int, confidence: float) -> bool:
    return rejection_count >= REJECTION_LIMIT or confidence < 0.3
