from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import pdfplumber


# ---------- Data model ----------

@dataclass
class CashFlowStatement:
    reporting_period: Optional[str]
    net_cash_operating: Optional[float]
    net_cash_investing: Optional[float]
    net_cash_financing: Optional[float]
    net_change_in_cash: Optional[float]
    cash_beginning: Optional[float]
    cash_ending: Optional[float]
    # All parsed line items: label -> numeric value
    line_items: Dict[str, float]
    # Optional: items grouped by section (operating / investing / financing)
    section_items: Dict[str, Dict[str, float]]


# ---------- Helpers ----------

def extract_text_from_pdf(path: str) -> str:
    """Extract all text from a PDF (simple concat of page texts)."""
    parts: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)
