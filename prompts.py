"""
prompts.py
----------
Holds the (short, stable) system prompt and the RAG prompt template.
Keeping this prompt compact and unchanged across requests is what keeps
token usage low (see cahier des charges, section 5 "Prompt Engineering
economique").
"""

SYSTEM_PROMPT = """You are CardioAssist, an assistant that provides general information about cardiovascular health.

Rules:
- Never provide a diagnosis.
- Never prescribe medication.
- Never recommend modifying or stopping a treatment.
- Use simple, short language.
- Indicate uncertainty when necessary.
- Do not present a hypothesis as a fact.
- If a situation seems urgent, recommend urgent medical help.
- You do not replace a healthcare professional."""


# Used only when RAG (Phase 2) is enabled and relevant passages were found.
# Kept intentionally short: at most a few passages are ever inserted here,
# never full documents (see section 9, "RAG economique").
RAG_CONTEXT_HEADER = "Relevant reference information (use only if helpful, do not quote at length):"


def build_prompt(history_text: str, question: str, context_passages=None) -> str:
    """
    Assembles the final prompt sent to the model.

    Format (section 6):
        SYSTEM PROMPT
        + USEFUL CONTEXT (optional, RAG)
        + CONVERSATION (last few turns only)
        + QUESTION

    We deliberately avoid sending the entire conversation or full documents.
    """
    parts = [SYSTEM_PROMPT]

    if context_passages:
        context_block = "\n".join(f"- {p}" for p in context_passages)
        parts.append(f"\n{RAG_CONTEXT_HEADER}\n{context_block}")

    if history_text:
        parts.append(f"\nConversation so far:\n{history_text}")

    parts.append(f"\nUser: {question}\nAssistant:")

    return "\n".join(parts)
