import sys
import logging
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.graph import build_graph

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Agent Orchestration System", layout="wide")

st.markdown("# Agent Orchestration System")
st.caption("Supervisor decomposes tasks, specialists execute, reviewer validates, humans approve when confidence is low.")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "result_state" not in st.session_state:
    st.session_state.result_state = None

with st.sidebar:
    st.markdown("### Task Input")
    question = st.text_area(
        "Question",
        placeholder="Compare the financial risk profile and regulatory exposure of Company X based on its latest 10-K filing and recent news coverage",
        height=120,
    )
    document_text = st.text_area(
        "Optional: paste document text (simulates uploaded 10-K excerpt)",
        height=150,
    )
    run = st.button("Run Task", type="primary", use_container_width=True)

if run and question.strip():
    with st.spinner("Supervisor decomposing task and dispatching specialists..."):
        initial_state = {
            "question": question.strip(),
            "document_text": document_text.strip(),
            "plan": {},
            "completed_outputs": {},
            "review_history": [],
            "needs_human_review": False,
            "human_approved": False,
            "final_answer": "",
            "status": "",
        }
        result = st.session_state.graph.invoke(initial_state)
        st.session_state.result_state = result

state = st.session_state.result_state

if state:
    st.markdown("## Execution Plan")
    for subtask in state["plan"]["subtasks"]:
        status_color = {"completed": "green", "rejected": "red", "pending": "gray"}.get(
            subtask["status"], "gray"
        )
        st.markdown(f"**[{subtask['assigned_to'].upper()}]** {subtask['description']}")
        st.markdown(f":{status_color}[{subtask['status']}] — confidence: {subtask['confidence']:.2f}")
        with st.expander(f"Output from {subtask['assigned_to']}"):
            st.write(subtask["output"])

    st.markdown("## Review Results")
    for review in state["review_history"]:
        icon = "✅" if review["approved"] else "❌"
        st.markdown(f"{icon} Subtask {review['subtask_id']} — approved: {review['approved']}, confidence: {review['confidence']:.2f}")
        if review["issues"]:
            issues_text = [str(i) if not isinstance(i, str) else i for i in review['issues']]
            st.warning(f"Issues: {', '.join(issues_text)}")

    if state["needs_human_review"]:
        st.markdown("## Human Review Required")
        st.warning("One or more outputs were rejected or low-confidence. Review and approve manually.")
        col1, col2 = st.columns(2)
        if col1.button("Approve and Finalize", type="primary"):
            from agents.graph import finalize_node
            finalized = finalize_node(state)
            st.session_state.result_state = finalized
            st.rerun()
        if col2.button("Reject Task"):
            st.error("Task rejected by human reviewer.")
    elif state["status"] == "completed":
        st.markdown("## Final Answer")
        st.success(state["final_answer"])
