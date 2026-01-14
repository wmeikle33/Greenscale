from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

@dataclass
class CashFlowStatement:
    reporting_period: Optional[str]
    net_cash_operating: Optional[float]
    net_cash_investing: Optional[float]
    net_cash_financing: Optional[float]
    net_change_in_cash: Optional[float]
    cash_beginning: Optional[float]
    cash_ending: Optional[float]
    line_items: Dict[str, float]
    section_items: Dict[str, Dict[str, float]]

