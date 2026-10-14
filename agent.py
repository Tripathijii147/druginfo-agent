"""LangGraph agent: emergency guardrail -> tool-calling LLM loop (RAG, interaction check, FAERS)."""
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import rag
from config import REFUSAL
from tools import TOOLS

SYSTEM = (
    "You are DrugInfo Agent. You answer questions about medicines using ONLY what your tools return.\n"
    "- Factual question about one drug -> call search_labels.\n"
    "- Can two drugs be taken together / interaction question -> call check_interaction.\n"
    "- Most reported real-world side effects -> call adverse_event_stats.\n"
    "- You may call several tools. Never answer from memory.\n"
    "- Cite every claim with the tag the tool gave, e.g. [warfarin | Drug Interactions].\n"
    f"- If the tool results do not answer the question, or it is not about medicines, reply exactly: {REFUSAL}\n"
    "- 'No mention in the label' is not proof that a combination is safe; say so.\n"
    "- Never give personal dosing or treatment advice. Keep answers short. Remind the user to ask a "
    "doctor or pharmacist for decisions about their own care."
)

EMERGENCY = re.compile(
    r"overdos|can'?t breathe|cannot breathe|chest pain|suicid|kill myself|want to die|self[- ]harm|"
    r"anaphyla|swollen (face|tongue|throat)|poison|seizure|unconscious", re.I)
EMERGENCY_REPLY = (
    "This could be an emergency. Please call your local emergency number now (112 in India, 911 in the US) "
    "or go to the nearest emergency room. If someone may have taken too much of a medicine, contact a "
    "poison control centre right away. I can't help with emergencies, and I'm not a substitute for a doctor.")


def make_graph(llm=None, checkpointer=None):
    llm_tools = (llm or rag.get_llm()).bind_tools(TOOLS)

    def emergency(state):
        return {"messages": [AIMessage(content=EMERGENCY_REPLY)]}

    def agent(state):
        msgs = [SystemMessage(content=SYSTEM)] + state["messages"]
        err = None
        for _ in range(2):  # tool-call formatting glitches from the model are usually transient
            try:
                return {"messages": [llm_tools.invoke(msgs)]}
            except Exception as e:  # noqa: BLE001
                err = e
        return {"messages": [AIMessage(content=f"The language model call failed ({type(err).__name__}). "
                                               "Please try again in a moment.")]}

    def entry(state):
        last = state["messages"][-1]
        return "emergency" if isinstance(last, HumanMessage) and EMERGENCY.search(last.content) else "agent"

    g = StateGraph(MessagesState)
    g.add_node("emergency", emergency)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_conditional_edges(START, entry, {"emergency": "emergency", "agent": "agent"})
    g.add_conditional_edges("agent", tools_condition)   # tool calls -> "tools", else END
    g.add_edge("tools", "agent")
    g.add_edge("emergency", END)
    return g.compile(checkpointer=checkpointer)


def verify_citations(answer_text, steps):
    """Verify that every citation tag [drug | section] in answer_text appears in tool outputs.

    Returns (valid, invalid_tags, valid_tags_found_in_tools).
    """
    cited_tags = set(re.findall(r"\[([a-zA-Z0-9_\- ]+\s*\|\s*[a-zA-Z0-9_\- ]+)\]", answer_text))
    if not cited_tags:
        return True, set(), set()

    tool_outputs = " ".join(s.get("output", "") for s in steps)
    available_tags = set(re.findall(r"\[([a-zA-Z0-9_\- ]+\s*\|\s*[a-zA-Z0-9_\- ]+)\]", tool_outputs))
    # Normalize comparison (case-insensitive and trimmed)
    norm_available = {"|".join(part.strip().lower() for part in t.split("|")): t for t in available_tags}

    invalid = set()
    for tag in cited_tags:
        norm_tag = "|".join(part.strip().lower() for part in tag.split("|"))
        if norm_tag not in norm_available:
            invalid.add(tag)

    return len(invalid) == 0, invalid, available_tags


def ask(graph, question, history=(), recursion_limit=10):
    """history: [(role, text)] with role 'user' or 'assistant'. Returns (final_text, steps)."""
    msgs = [HumanMessage(content=t) if r == "user" else AIMessage(content=t) for r, t in history]
    msgs.append(HumanMessage(content=question))
    try:
        out = graph.invoke({"messages": msgs}, config={"recursion_limit": recursion_limit})
    except GraphRecursionError:
        return "The agent took too many steps. Try a simpler question.", []
    new = out["messages"][len(msgs):]
    steps, order = {}, []
    for m in new:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                steps[tc["id"]] = {"tool": tc["name"], "args": tc["args"], "output": ""}
                order.append(tc["id"])
        elif isinstance(m, ToolMessage) and m.tool_call_id in steps:
            steps[m.tool_call_id]["output"] = str(m.content)
    final = str(new[-1].content) if new else ""
    step_list = [steps[i] for i in order]

    # Citation Verification: If the agent used tools and generated answers with citations,
    # verify that cited citations match real tool outputs.
    if step_list and final:
        valid, invalid_tags, available_tags = verify_citations(final, step_list)
        if not valid and available_tags:
            # Re-prompt once to correct hallucinated citation tags
            correct_prompt = (
                f"Your previous answer included citations that were not in your tool outputs: {', '.join(sorted(invalid_tags))}.\n"
                f"Only use citations that actually appeared in the tool excerpts: {', '.join(sorted(available_tags))}.\n"
                "Please rewrite your answer keeping only verified citations."
            )
            retry_msgs = list(out["messages"]) + [HumanMessage(content=correct_prompt)]
            try:
                retry_out = graph.invoke({"messages": retry_msgs}, config={"recursion_limit": 5})
                final_retry = retry_out["messages"][-1]
                if isinstance(final_retry, AIMessage) and final_retry.content:
                    final = str(final_retry.content)
            except Exception:
                pass

    return final, step_list

