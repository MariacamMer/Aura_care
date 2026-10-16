"""Prueba rapida de Gemini (1 llamada). Uso:
    export GEMINI_API_KEY="tu_clave"          # Windows PowerShell:  $env:GEMINI_API_KEY="tu_clave"
    export AURA_DATA_DIR="../datos"           # carpeta donde estan D1..D7
    python probar_llm.py
"""
import os, sys
os.environ.setdefault("AURA_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "datos"))
key = os.environ.get("GEMINI_API_KEY")
if not key: sys.exit("Falta GEMINI_API_KEY (ver instrucciones arriba).")
os.environ.update(AURA_LLM_BACKEND="openai_compat", AURA_LLM_KEY=key,
                  AURA_LLM_URL="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                  AURA_LLM_MODEL=os.environ.get("AURA_LLM_MODEL", "gemini-3.8-flash"))
import aura_care as A
S = A.load_d6()
r = A.respond("Llevo semanas con muchas evaluaciones y no sé a quién acudir. Prefiero algo digital.", S, country="Chile")
print("Respondió:", r["llm"]); print("Error:", r["llm_error"]); print("Validación:", r["problems"] or "OK"); print("\n", r["answer"])
print("\n>>> RESULTADO:", "Gemini funciona" if r["llm"] == "openai_compat" else "NO se uso Gemini (revisar clave/modelo/internet)")
