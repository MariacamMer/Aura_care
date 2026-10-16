"""Score preventivo NO clinico (agregado). Reglas aditivas y transparentes sobre D1. No es un diagnostico."""
import pandas as pd

COMPONENTES = {  # variable D1 -> puntos
    "anxiety_band":  {"low": 0, "mild": 1, "moderate": 2, "high": 3},
    "stress_band":   {"low": 0, "moderate": 1, "high": 2},
    "support_network": {"adequate": 0, "limited": 1, "none": 2},
}
def score_preventivo(d1: pd.DataFrame) -> pd.Series:
    s = sum(d1[c].map(m) for c, m in COMPONENTES.items())
    s += (d1.evaluation_overload == "yes") * 1
    s += (d1.sleep_hours < 6.5) * 1
    s += (d1.services_awareness == "unaware") * 1
    s += (d1.employability_concern == "high") * 1
    return s.astype(int)

def nivel(s: pd.Series) -> pd.Series:
    return pd.cut(s, [-1, 3, 6, 99], labels=["bajo", "medio", "alto"])
