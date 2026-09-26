"""
safety.py
---------
A small, deliberately simple safety filter that runs BEFORE the LLM is
called (cahier des charges, section 4). It is keyword/pattern based, not a
learned classifier — the spec explicitly says the MVP filter should stay
simple and can be improved later.

check_message() returns a dict:
    {"status": "ok" | "therapeutic" | "urgent" | "injection", "message": str | None}

If status != "ok", the caller must NOT call the LLM and must instead show
`message` directly to the user.
"""

import re

# ---------------------------------------------------------------------------
# 4.1 Therapeutic requests (stop / change / double a treatment or dose)
# ---------------------------------------------------------------------------
THERAPEUTIC_PATTERNS = [
    r"\bstop\b.{0,25}\b(medication|treatment|drug|pills?|meds?)\b",
    r"\b(medication|treatment|drug|pills?|meds?)\b.{0,25}\bstop\b",
    r"\bdouble\b.{0,15}\bdos(e|age)\b",
    r"\bincrease\b.{0,15}\bdos(e|age)\b",
    r"\bdecrease\b.{0,15}\bdos(e|age)\b",
    r"\bskip\b.{0,15}\b(dose|medication|pill|meds?)\b",
    r"\breplace\b.{0,25}\b(medication|treatment|drug)\b",
    r"\bchange\b.{0,25}\b(medication|treatment|dosage|dose)\b",
    r"\bcan i (stop|quit)\b.{0,25}\b(medication|treatment|pills?|meds?)\b",
]

THERAPEUTIC_MESSAGE = (
    "I can't advise you to change, stop, or adjust a treatment or dose. "
    "Please ask your doctor or pharmacist for guidance."
)

# ---------------------------------------------------------------------------
# 4.2 Potentially urgent situations
# ---------------------------------------------------------------------------
URGENT_PATTERNS = [
    r"\bchest pain\b",
    r"\btightness in (my|the) chest\b",
    r"\bcan'?t breathe\b",
    r"\bcannot breathe\b",
    r"\bdifficulty breathing\b",
    r"\bshortness of breath\b",
    r"\bheart attack\b",
    r"\bstroke\b",
    r"\bnumbness (in|on) (my|the)? ?(arm|face|leg)\b",
    r"\bslurred speech\b",
    r"\bfaint(ed|ing)?\b",
    r"\bloss of consciousness\b",
    r"\blosing consciousness\b",
    r"\bblue lips\b",
    r"\bcollapsed\b",
    r"\bsevere (chest|arm|jaw) pain\b",
    r"\bracing heart\b.{0,20}\b(dizzy|faint|pain)\b",
]

URGENT_MESSAGE = (
    "The symptoms you describe may require urgent medical evaluation. "
    "Do not rely only on this assistant — contact your local emergency "
    "services immediately or seek urgent medical care."
)

# ---------------------------------------------------------------------------
# Prompt-injection / rule-override attempts (test 4 in the cahier des charges)
# ---------------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"\bignore\b.{0,25}\b(previous|all|prior)\b.{0,15}\b(instructions|rules)\b",
    r"\bforget\b.{0,15}\b(instructions|rules|guidelines)\b",
    r"\bact as\b.{0,20}\b(doctor|physician|unfiltered|dan)\b",
    r"\bjailbreak\b",
    r"\bbypass\b.{0,15}\b(rules|safety|filter)\b",
    r"\byou are no longer\b",
    r"\bpretend (you are|to be)\b.{0,20}\b(doctor|unrestricted)\b",
    r"\bgive me a diagnosis\b",
    r"\bdiagnose me\b",
]

INJECTION_MESSAGE = (
    "I can't set aside my safety guidelines. I can only provide general "
    "cardiovascular health information — I can't diagnose, prescribe, or "
    "advise changes to a treatment."
)


def _matches_any(text: str, patterns) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def check_message(text: str) -> dict:
    """
    Runs the safety checks in priority order:
    1) urgent situations (highest priority — could be life-threatening)
    2) therapeutic requests
    3) prompt-injection attempts
    Returns {"status": ..., "message": ...} — status "ok" means the LLM
    may be called normally.
    """
    if not text or not text.strip():
        return {"status": "ok", "message": None}

    if _matches_any(text, URGENT_PATTERNS):
        return {"status": "urgent", "message": URGENT_MESSAGE}

    if _matches_any(text, THERAPEUTIC_PATTERNS):
        return {"status": "therapeutic", "message": THERAPEUTIC_MESSAGE}

    if _matches_any(text, INJECTION_PATTERNS):
        return {"status": "injection", "message": INJECTION_MESSAGE}

    return {"status": "ok", "message": None}
