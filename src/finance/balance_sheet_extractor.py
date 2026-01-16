from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import pdfplumber


@dataclass
class BalanceSheet:
    reporting_date: Optional[str]
    total_assets: Optional[float]
    total_liabilities: Optional[float]
    total_equity: Optional[float]
    # raw line items if you want more detail
    line_items: Dict[str, float]

def extract_text_from_pdf(path: str) -> str:
    parts: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)
