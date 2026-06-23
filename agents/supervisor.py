import json
import re
import requests
import logging

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"


def llm_call(prompt: str, system: str = "", max_tokens: int = 512) -> str:
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.1, "num_ctx": 2048},
        }, timeout=30)
        return r.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return ""


def decompose_task(question: str) -> list[dict]:
    prompt = f"""You are a task planning supervisor. Break this question into subtasks.

Question: {question}

Assign each subtask to exactly one of: research (web/news search), document (extracts facts from uploaded documents), synthesis (combines findings into a final answer).

Reply ONLY with a JSON array like:
[{{"id": "1", "description": "...", "assigned_to": "research"}}, {{"id": "2", "description": "...", "assigned_to": "document"}}, {{"id": "3", "description": "...", "assigned_to": "synthesis"}}]"""

    resp = llm_call(prompt, max_tokens=300)
    try:
        match = re.search(r'\[.*\]', resp, re.DOTALL)
        if match:
            tasks = json.loads(match.group())
            for t in tasks:
                t["status"] = "pending"
                t["output"] = ""
                t["confidence"] = 0.0
            return tasks
    except Exception as e:
        logger.warning(f"Decomposition parse failed: {e}")

    return [
        {"id": "1", "description": f"Research recent news related to: {question}",
         "assigned_to": "research", "status": "pending", "output": "", "confidence": 0.0},
        {"id": "2", "description": f"Extract relevant facts from uploaded documents for: {question}",
         "assigned_to": "document", "status": "pending", "output": "", "confidence": 0.0},
        {"id": "3", "description": f"Synthesize findings into a final answer for: {question}",
         "assigned_to": "synthesis", "status": "pending", "output": "", "confidence": 0.0},
    ]
