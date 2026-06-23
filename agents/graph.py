import logging
from langgraph.graph import StateGraph, END

from agents.state import GraphState
from agents.supervisor import decompose_task
from agents.specialists import research_agent, document_agent, synthesis_agent
from agents.reviewer import review_output, should_escalate_to_human
from memory.memory_store import WorkingMemory, LongTermMemory

logger = logging.getLogger(__name__)

working_memory = WorkingMemory()
long_term_memory = LongTermMemory()


def supervisor_node(state: GraphState) -> GraphState:
    task_id = state["question"][:50]
    working_memory.init_task(task_id, state["question"])

    similar = long_term_memory.retrieve_similar(state["question"], n=2)
    if similar:
        logger.info(f"Found {len(similar)} similar past tasks in long-term memory")

    subtasks = decompose_task(state["question"])
    state["plan"] = {"question": state["question"], "subtasks": subtasks}
    state["completed_outputs"] = {}
    state["review_history"] = []
    state["needs_human_review"] = False
    state["status"] = "planned"
    return state


def specialist_dispatch_node(state: GraphState) -> GraphState:
    document_text = state.get("document_text", "")
    completed = state["completed_outputs"]

    for subtask in state["plan"]["subtasks"]:
        role = subtask["assigned_to"]
        desc = subtask["description"]

        if role == "research":
            result = research_agent(desc, state["question"])
        elif role == "document":
            result = document_agent(desc, state["question"], document_text)
        elif role == "synthesis":
            non_synthesis = {k: v for k, v in completed.items() if k != "synthesis"}
            result = synthesis_agent(desc, state["question"], non_synthesis)
        else:
            result = {"output": "Unknown agent type", "confidence": 0.0, "sources": []}

        subtask["output"] = result["output"]
        subtask["confidence"] = result["confidence"]
        completed[role] = result["output"]

    state["completed_outputs"] = completed
    state["status"] = "specialists_complete"
    return state


def review_node(state: GraphState) -> GraphState:
    reviews = []
    needs_escalation = False

    for subtask in state["plan"]["subtasks"]:
        source_context = "\n".join(
            f"{k}: {v}" for k, v in state["completed_outputs"].items() if k != subtask["assigned_to"]
        )
        prior = next((r["rejection_count"] for r in state["review_history"]
                      if r.get("subtask_id") == subtask["id"]), 0)

        review = review_output(
            subtask["description"], subtask["output"], source_context,
            subtask["confidence"], prior_rejections=prior,
        )
        review["subtask_id"] = subtask["id"]
        reviews.append(review)

        if not review["approved"]:
            subtask["status"] = "rejected"
        else:
            subtask["status"] = "completed"

        if should_escalate_to_human(review["rejection_count"], review["confidence"]):
            needs_escalation = True

    state["review_history"] = reviews
    state["needs_human_review"] = needs_escalation
    state["status"] = "reviewed"
    return state


def finalize_node(state: GraphState) -> GraphState:
    synthesis_output = state["completed_outputs"].get("synthesis", "")
    state["final_answer"] = synthesis_output or "No synthesis output produced."
    state["status"] = "completed"

    all_approved = all(r["approved"] for r in state["review_history"])
    long_term_memory.store_outcome(
        question=state["question"],
        plan_summary=str([t["description"] for t in state["plan"]["subtasks"]]),
        outcome=state["final_answer"],
        success=all_approved,
    )
    return state


def route_after_review(state: GraphState) -> str:
    if state["needs_human_review"]:
        return "human_review"
    return "finalize"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("specialists", specialist_dispatch_node)
    graph.add_node("review", review_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "specialists")
    graph.add_edge("specialists", "review")
    graph.add_conditional_edges(
        "review", route_after_review,
        {"human_review": END, "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()
