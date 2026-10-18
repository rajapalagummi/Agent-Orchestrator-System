import json
import logging
from agents.graph import build_graph

logging.basicConfig(level=logging.WARNING)

QUESTIONS = [
    "What are Apple's latest quarterly revenue figures?",
    "Summarize Tesla's risk factors from their most recent 10-K.",
    "What is Microsoft's cloud revenue growth year over year?",
    "What did the Federal Reserve say about interest rates in 2024?",
    "Explain Amazon's operating income trends over the past two years.",
    "What are the main business segments of Alphabet Inc?",
    "What acquisitions did Meta make in the last 12 months?",
    "How does NVIDIA's data center revenue compare to gaming revenue?",
    "What are the key risk factors for JPMorgan Chase?",
    "Summarize the competitive landscape for cloud storage providers.",
    "What is OpenAI's reported valuation and funding history?",
    "How has Netflix subscriber growth trended recently?",
    "What are the regulatory risks facing large US tech companies?",
    "Describe Boeing's recent financial performance and key challenges.",
    "What are Pfizer's main revenue drivers post-pandemic?",
    "How does Berkshire Hathaway allocate its investment portfolio?",
    "What are the main segments of Goldman Sachs's business?",
    "Describe the competitive dynamics in the semiconductor industry.",
    "What ESG commitments has ExxonMobil made recently?",
    "Summarize the key findings from a recent AI industry report.",
]

def run_batch():
    graph = build_graph()
    total_subtasks = 0
    approved = 0
    rejected = 0
    escalated = 0
    results = []

    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i:02d}/20] {q[:60]}...")
        state = {
            "question": q,
            "document_text": "",
            "plan": {},
            "completed_outputs": {},
            "review_history": [],
            "needs_human_review": False,
            "human_approved": False,
            "final_answer": "",
            "status": "",
        }
        try:
            out = graph.invoke(state)
            for r in out.get("review_history", []):
                total_subtasks += 1
                if r.get("approved"):
                    approved += 1
                else:
                    rejected += 1
            if out.get("needs_human_review"):
                escalated += 1
            results.append({
                "question": q,
                "status": out.get("status"),
                "needs_human_review": out.get("needs_human_review"),
                "subtask_reviews": out.get("review_history", []),
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"question": q, "error": str(e)})

    reject_pct = round(rejected / total_subtasks * 100, 1) if total_subtasks else 0
    summary = {
        "total_questions": len(QUESTIONS),
        "total_subtasks_reviewed": total_subtasks,
        "approved": approved,
        "rejected": rejected,
        "escalated_to_human": escalated,
        "rejection_rate_pct": reject_pct,
    }
    print("\n=== RESULTS ===")
    print(json.dumps(summary, indent=2))
    with open("eval_results.json", "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)
    print("\nSaved to eval_results.json")
    return summary

if __name__ == "__main__":
    run_batch()
