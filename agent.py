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
    "- Can two drugs be taken together / pairwise interaction question -> call check_interaction.\n"
    "- Multiple medications list / 3 or more drugs polypharmacy audit -> call check_medication_list.\n"
    "- Most reported real-world side effects -> call adverse_event_stats.\n"
    "- You may call several tools. Never answer from memory.\n"
    "- Cite every claim with the tag the tool gave, e.g. [warfarin | Drug Interactions].\n"
    f"- If the tool results do not answer the question, or it is not about medicines, reply exactly: {REFUSAL}\n"
    "- 'No mention in the label' is not proof that a combination is safe; say so.\n"
    "- Never give personal dosing or treatment advice. Keep answers short. Remind the user to ask a "
    "doctor or pharmacist for decisions about their own care."
)

# Safety Guardrail RegExes
EMERGENCY = re.compile(
    r"overdos|can'?t breathe|cannot breathe|chest pain|suicid|kill myself|want to die|self[- ]harm|"
    r"anaphyla|swollen (face|tongue|throat)|poison|seizure|unconscious", re.I)
EMERGENCY_REPLY = (
    "This could be an emergency. Please call your local emergency number now (112 in India, 911 in the US) "
    "or go to the nearest emergency room. If someone may have taken too much of a medicine, contact a "
    "poison control centre right away. I can't help with emergencies, and I'm not a substitute for a doctor.")

PROMPT_INJECTION = re.compile(
    r"(ignore\s+(previous|all)\s+instructions|system\s+prompt|jailbreak|disregard|you\s+are\s+now|DAN\s+mode|"
    r"developer\s+mode|override\s+safety|bypass\s+rules)", re.I)
INJECTION_REPLY = "I cannot fulfill this request. I am only permitted to provide factual information from verified FDA drug labels."

PERSONAL_DOSAGE = re.compile(
    r"(what\s+dose\s+should\s+i\s+take|how\s+much\s+(should\s+i|can\s+i)\s+take|my\s+dose|prescribe\s+me|"
    r"can\s+i\s+take\s+\d+\s*mg|recommend\s+a\s+dose\s+for\s+me)", re.I)
DOSAGE_DEFLECTION = (
    "I cannot recommend personal dosages or give treatment advice. Dosage depends on individual patient factors, "
    "medical history, and kidney/liver function. Please consult a licensed physician or pharmacist for your specific dosage.")

PEDIATRIC_QUERY = re.compile(
    r"(for\s+my\s+(baby|infant|toddler|child|newborn|\d+[- ]year[- ]old|kid)|"
    r"(give|administer).*(to\s+my\s+|\b)(baby|child|infant|toddler|newborn|\d+[- ]year[- ]old|kid)|"
    r"can\s+i\s+give\s+(a\s+)?(baby|child|infant|toddler))", re.I)
PEDIATRIC_DEFLECTION = (
    "Pediatric drug administration requires strict weight-based calculation and direct medical supervision. "
    "I cannot advise on giving medicines to children or infants. Please contact a pediatrician or pharmacist.")


def make_graph(llm=None, checkpointer=None):
    llm_tools = (llm or rag.get_llm()).bind_tools(TOOLS)

    def emergency_node(state):
        return {"messages": [AIMessage(content=EMERGENCY_REPLY)]}

    def injection_node(state):
        return {"messages": [AIMessage(content=INJECTION_REPLY)]}

    def dosage_node(state):
        return {"messages": [AIMessage(content=DOSAGE_DEFLECTION)]}

    def pediatric_node(state):
        return {"messages": [AIMessage(content=PEDIATRIC_DEFLECTION)]}

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
        if isinstance(last, HumanMessage):
            txt = last.content
            if EMERGENCY.search(txt):
                return "emergency"
            if PROMPT_INJECTION.search(txt):
                return "injection"
            if PERSONAL_DOSAGE.search(txt):
                return "dosage"
            if PEDIATRIC_QUERY.search(txt):
                return "pediatric"
        return "agent"

    g = StateGraph(MessagesState)
    g.add_node("emergency", emergency_node)
    g.add_node("injection", injection_node)
    g.add_node("dosage", dosage_node)
    g.add_node("pediatric", pediatric_node)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))

    g.add_conditional_edges(
        START,
        entry,
        {
            "emergency": "emergency",
            "injection": "injection",
            "dosage": "dosage",
            "pediatric": "pediatric",
            "agent": "agent"
        }
    )
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    g.add_edge("emergency", END)
    g.add_edge("injection", END)
    g.add_edge("dosage", END)
    g.add_edge("pediatric", END)
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


REASSURANCE_PATTERNS = re.compile(
    r"\b(is safe|are safe|can be safely taken|no known interaction|completely safe|safe to take together|fine to take together|no risk|no problem)\b",
    re.I
)


def post_check_safety(final_text, steps):
    """Safety guardrail: ensures no draft reassures about an unverified or flagged interaction pair.
    If tools flagged an interaction or reported that an interaction could not be verified in the package insert,
    any draft containing false reassurance ('is safe to take', 'no problem', etc.) is corrected."""
    if not steps or not final_text:
        return final_text

    tool_outputs = " ".join(s.get("output", "") for s in steps)
    has_flag = "at least one label flags" in tool_outputs or "Flagged Interactions" in tool_outputs
    has_unverified = "could not verify interaction" in tool_outputs or "NOT proof of safety" in tool_outputs

    if (has_flag or has_unverified) and REASSURANCE_PATTERNS.search(final_text):
        if has_flag:
            return (
                "Warning: A potential interaction or dosage limitation was identified in official FDA package inserts.\n\n"
                f"{final_text}\n\n"
                "Please do not assume this combination is safe. Consult your physician or pharmacist before combining these medications."
            )
        elif has_unverified:
            return (
                "Caution: Interaction data between these medications could not be verified in the indexed FDA label sections.\n\n"
                "Absence of an explicit warning in a drug label is NOT proof of safety. Please consult a doctor or pharmacist for clinical guidance."
            )

    return final_text


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

    # Safety Post-Check: override any false reassurance on unverified or flagged pairs
    final = post_check_safety(final, step_list)

    return final, step_list


