import json
import re
import logging

from agents.supervisor import llm_call
from tools.registry import search_sec_edgar, search_news, extract_document_facts

logger = logging.getLogger(__name__)


def research_agent(subtask_description: str, question: str) -> dict:
    sec_results = search_sec_edgar(question, limit=2)
    news_results = search_news(question, limit=2)
    all_results = sec_results + news_results

    if not all_results:
        return {"output": "No research results found.", "confidence": 0.2, "sources": []}

    context = "\n".join([f"[{r['source']}] {r['text']}" for r in all_results])
    prompt = f"""Task: {subtask_description}

Research findings:
{context}

Summarize the relevant findings in 2-3 sentences. Then on a new line write CONFIDENCE: followed by a number 1-5 reflecting how well these sources actually answer the task."""

    resp = llm_call(prompt, max_tokens=300)
    confidence = _extract_confidence(resp)
    summary = re.sub(r'CONFIDENCE:\s*\d', '', resp).strip()

    return {
        "output": summary or "Unable to summarize research findings.",
        "confidence": confidence,
        "sources": [r["source"] for r in all_results],
    }


def document_agent(subtask_description: str, question: str, document_text: str = "") -> dict:
    if not document_text.strip():
        return research_agent(subtask_description, question)

    facts = extract_document_facts(document_text)
    prompt = f"""Task: {subtask_description}

Document content:
{facts}

Extract the facts relevant to this task in 2-3 sentences. Then on a new line write CONFIDENCE: followed by a number 1-5 reflecting how directly the document addresses the task."""

    resp = llm_call(prompt, max_tokens=300)
    confidence = _extract_confidence(resp)
    summary = re.sub(r'CONFIDENCE:\s*\d', '', resp).strip()

    return {
        "output": summary or "Unable to extract document facts.",
        "confidence": confidence,
        "sources": ["uploaded document"],
    }


def synthesis_agent(subtask_description: str, question: str, prior_outputs: dict) -> dict:
    context = "\n\n".join([f"[{k}]: {v}" for k, v in prior_outputs.items()])
    prompt = f"""Task: {subtask_description}
Original question: {question}

Findings from other agents:
{context}

Synthesize a final answer using ONLY the findings above. Do not invent facts not present in the findings. Then on a new line write CONFIDENCE: followed by a number 1-5 reflecting how well-supported your synthesis is by the findings."""

    resp = llm_call(prompt, max_tokens=400)
    confidence = _extract_confidence(resp)
    summary = re.sub(r'CONFIDENCE:\s*\d', '', resp).strip()

    return {
        "output": summary or "Unable to synthesize findings.",
        "confidence": confidence,
        "sources": list(prior_outputs.keys()),
    }


def _extract_confidence(text: str) -> float:
    match = re.search(r'CONFIDENCE:\s*(\d)', text)
    if match:
        return min(5, max(1, int(match.group(1)))) / 5
    return 0.5
