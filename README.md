# Agent Orchestration System

A multi-agent system where a Supervisor decomposes a complex question into subtasks, delegates to specialized tool-using agents, a Reviewer validates outputs against source material before they're accepted, and low-confidence or repeatedly rejected outputs escalate to a human approval step.

## Architecture

```
Question
    |
SUPERVISOR
  Checks long-term memory for similar past tasks
  Decomposes question into subtasks (research / document / synthesis)
    |
SPECIALISTS (dispatched per subtask)
  Research agent -> SEC EDGAR + news search
  Document agent -> extracts facts from provided document text
  Synthesis agent -> combines other specialists' outputs
  Each specialist self-reports a confidence score (1-5)
    |
REVIEWER
  Checks confidence threshold
  Checks output against source material for unsupported claims
  Approves or rejects, tracks rejection count per subtask
    |
ROUTING
  If rejection count >= 2 or confidence < 0.3 -> HUMAN REVIEW
  Otherwise -> FINALIZE
    |
FINALIZE
  Stores outcome in long-term memory (ChromaDB) for future similar tasks
    |
Output: Final synthesized answer, full agent trace, review history
```

## Why this is different from a sequential pipeline

Each agent is an independent LangGraph node with its own role, prompt, and inputs — not a Python loop calling functions in sequence. The graph has conditional edges: a rejected or low-confidence output routes to human review instead of finalizing silently. Long-term memory persists across tasks in ChromaDB, so the system can reference what worked on similar prior questions.

## Stack

- **Orchestration** - LangGraph (state machine, conditional edges)
- **LLM** - Phi-3 Mini via Ollama (local, no API key required)
- **Long-term memory** - ChromaDB (semantic recall of prior task outcomes)
- **Short-term memory** - in-memory working store scoped per task
- **Tools** - SEC EDGAR API, Wikipedia/news search
- **UI** - Streamlit

## Setup

Prerequisites: Python 3.12, Docker Desktop, Ollama

```
# Step 1 - Install Ollama and pull model
brew install ollama
ollama serve
ollama pull phi3:mini

# Step 2 - Start ChromaDB
docker run -d -p 8000:8000 chromadb/chroma

# Step 3 - Create environment
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Step 4 - Run
python3 -m streamlit run ui/app.py
```

## Example test case

Question: "Compare the financial risk profile and regulatory exposure of Company X based on its latest 10-K filing and recent news coverage"

Paste a sample 10-K excerpt into the document field to test the document agent. Try removing the document text to see the document agent self-report low confidence and correctly flag nothing to extract.

## Privacy

- No telemetry or external data collection beyond the public APIs queried
- Long-term memory stored locally in ChromaDB
