"""AURA Care - nucleo del agente (orientacion preventiva NO clinica).

Capas:
  1. Seguridad determinista (crisis) -> corre ANTES del LLM y puede anularlo.
  2. Necesidad (reglas simples sobre texto del estudiante).
  3. RAG sobre D6: recupera servicios reales (service_id) con razon explicita.
  4. LLM generativo (adaptador): redacta la respuesta SOLO con lo recuperado.
  5. Validador de salida: bloquea servicios inventados y lenguaje diagnostico.
"""
import hashlib, json, os, re, time, unicodedata, urllib.error, urllib.request

VERSION = "2026-10-11-g"
CACHE_HIT = False   # se imprime en el notebook para confirmar que se cargo el archivo nuevo
DATA_DIR = os.environ.get("AURA_DATA_DIR", "/mnt/user-data/uploads")

# ---------- Carga de datos ----------
def load_d6(path=None):
    g = json.load(open(path or f"{DATA_DIR}/D6_services_map.geojson", encoding="utf-8"))
    return {f["properties"]["service_id"]: f["properties"] for f in g["features"]}

def load_json(name):
    return json.load(open(f"{DATA_DIR}/{name}", encoding="utf-8"))

# ---------- Utilidades ----------
def norm(t):
    t = unicodedata.normalize("NFD", t.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")

# ---------- 1. Seguridad: crisis ----------
# Lexico DISENADO por el equipo (D4/D5 no contienen mensajes de crisis: se verifico 0 coincidencias).
CRISIS_PATTERNS = [
    r"suicid", r"quitarme la vida", r"quitarse la vida", r"acabar con (mi vida|todo)",
    r"no quiero (seguir )?viv", r"quiero morir", r"me quiero morir", r"mejor (estar|estaria) muert",
    r"hacerme dano", r"lastimarme", r"autolesi", r"cortarme", r"me estoy cortando", r"cortandome", r"no aguanto mas", r"no vale la pena vivir",
    r"desaparecer para siempre", r"terminar con todo",
]
CRISIS_RE = re.compile("|".join(CRISIS_PATTERNS))

# Verificar antes de la demo (fuentes publicas consultadas en oct-2026).
EMERGENCY = {
    "Chile": "Línea de Prevención del Suicidio *4141 (24/7, gratuita) o SAMU 131 si hay riesgo inmediato.",
    "Perú": "Línea 113, opción 5 (salud mental, 24/7, gratuita) o SAMU 106 si hay riesgo inmediato.",
}

def detect_crisis(text):
    return bool(CRISIS_RE.search(norm(text)))

def crisis_response(country="Chile"):
    country = {"Peru": "Perú"}.get(country, country)
    line = EMERGENCY.get(country, " ".join(EMERGENCY.values()))
    return ("Siento mucho que estés pasando por esto, y gracias por decirlo. Soy una IA y no puedo acompañarte "
            "en una situación así: lo más importante ahora es que hables con una persona. "
            f"{line} Si puedes, avisa también a alguien de confianza que esté cerca. "
            "Si quieres, también puedo indicarte cómo contactar a tu unidad de bienestar universitario.")

# ---------- 2. Necesidad ----------
NEED_RULES = {  # motivos de D5 (motive) -> pistas lexicas; D4 agrega work_study_balance
    "academic_pressure": ["evaluacion", "examen", "prueba", "entrega", "nota", "rendimiento", "sobrecarga", "estres", "ansied"],
    "sleep_and_routine": ["dormi", "sueno", "descanso", "rutina", "insomnio", "cansad"],
    "social_support": ["solo", "sola", "red cercana", "amigos", "lejos de mi", "aislad", "pares"],
    "career_concern": ["empleo", "trabajo", "egreso", "carrera", "practica", "laboral", "titul"],
    "service_navigation": ["no se a quien", "no conozco", "donde pedir", "servicio", "canales", "por donde empezar"],
}
def classify_need(text):
    t = norm(text)
    scores = {k: sum(w in t for w in ws) for k, ws in NEED_RULES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "service_navigation"

_CLF = None
def train_need_classifier(d5):
    """TF-IDF + regresion logistica entrenada con D5 (mensajes del estudiante -> motive)."""
    global _CLF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    X = [" ".join(m["text"] for m in c["messages"] if m["role"] == "student") for c in d5]
    _CLF = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=1000))
    _CLF.fit(X, [c["motive"] for c in d5])
    return _CLF

def classify_need_ml(text):
    """Reglas primero (texto libre natural); si ninguna pista coincide, modelo entrenado en D5."""
    t = norm(text)
    if any(w in t for ws in NEED_RULES.values() for w in ws):
        return classify_need(text)
    return _CLF.predict([text])[0] if _CLF is not None else "service_navigation"

# Mapa necesidad -> tipos de servicio D6. DECISION DE DISENO del equipo:
# en D5 el 'referral' es independiente del 'motive' (reparto ~uniforme), por lo que D5 no sirve como verdad-terreno.
NEED_TO_TYPES = {
    "academic_pressure": ["counseling", "peer_support"],
    "sleep_and_routine": ["counseling", "peer_support"],
    "social_support": ["peer_support", "counseling"],
    "career_concern": ["career_guidance", "counseling"],
    "service_navigation": ["counseling", "peer_support", "career_guidance"],
}

def extract_prefs(text):
    t = norm(text)
    p = {}
    if any(w in t for w in ["digital", "online", "en linea", "por escrito", "chat"]): p["channel"] = "digital"
    elif any(w in t for w in ["telefon", "llamada", "llamar"]): p["channel"] = "phone"
    elif any(w in t for w in ["presencial", "en persona"]): p["channel"] = "in_person"
    if any(w in t for w in ["tarde", "noche", "evening", "despues de clases", "trabajo"]): p["evening"] = True
    return p

# ---------- 3. RAG sobre D6 ----------
def _closes_late(schedule):
    m = re.findall(r"(\d{2}):(\d{2})-(\d{2}):(\d{2})", schedule)
    return bool(m) and int(m[0][2]) >= 19

def retrieve(services, need, prefs=None, district=None, k=3):
    prefs = prefs or {}
    types = NEED_TO_TYPES[need]
    out = []
    for sid, s in services.items():
        if s["service_type"] not in types: continue
        score, why = 10 - 3 * types.index(s["service_type"]), [f"tipo {s['service_type']} acorde a la necesidad '{need}'"]
        if prefs.get("channel") and prefs["channel"] in s["channels"]:
            score += 3; why.append(f"ofrece canal {prefs['channel']}")
        elif prefs.get("channel"): score -= 2
        if prefs.get("evening") and _closes_late(s["schedule"]):
            score += 4; why.append(f"horario extendido ({s['schedule']})")
        if district and s["district_id"] == district:
            score += 2; why.append(f"en tu distrito ({district})")
        out.append({"service_id": sid, "name": s["name"], "type": s["service_type"], "district": s["district_id"],
                    "schedule": s["schedule"], "channels": s["channels"], "capacity": s["capacity"],
                    "eligibility": s["eligibility"], "score": score, "why": why, "why_pref": why[1:]})
    return sorted(out, key=lambda x: (-x["score"], x["service_id"]))[:k]

# ---------- 4. LLM (adaptador) ----------
SYSTEM_PROMPT = """Eres AURA Care, un asistente de IA de orientacion preventiva para estudiantes universitarios (datos sinteticos de Aethera).
REGLAS: (1) Di al inicio que eres una IA. (2) NO diagnostiques, no interpretes sintomas ni sugieras tratamientos.
(3) Recomienda SOLO los servicios de la lista RECURSOS (usa su service_id exacto) y explica por que. No inventes servicios, horarios ni requisitos.
(4) De cada servicio solo puedes decir lo que figura en RECURSOS (tipo, horario, canales, distrito). NO afirmes que un servicio sea 'para' un tema (presion academica, estres, ansiedad, sueno, empleo...): describelo solo por su tipo, horario, canales y distrito, y relaciona la recomendacion con lo que el estudiante pidio (canal, horario). (5) Maximo 120 palabras, tono cercano y respetuoso, sin estigmatizar. (6) Cierra recordando que la decision es del estudiante y que puede hablar con una persona del equipo de bienestar."""

DIAS = {"Mon-Fri": "lunes a viernes", "Mon-Sat": "lunes a sábado", "Mon-Thu": "lunes a jueves", "Tue-Sat": "martes a sábado"}
TIPOS = {"counseling": "consejería", "peer_support": "apoyo entre pares", "career_guidance": "orientación laboral"}
def build_prompt(user_text, resources, feedback=None):
    def sched(x):
        for k, v in DIAS.items(): x = x.replace(k, v)
        return x
    res = "\n".join(f"- {r['service_id']} | {r['name']} | tipo: {TIPOS.get(r['type'], r['type'])} | horario: {sched(r['schedule'])} | canales: {', '.join(r['channels'])} | "
                    f"coincide con lo que pidió: {'; '.join(r['why_pref']) or 'ninguna preferencia específica'}" for r in resources)
    extra = ""
    if feedback:
        extra = ("\n\nCORRECCION: tu respuesta anterior incumplio estas reglas: " + "; ".join(feedback) +
                 ". Reescribela describiendo cada servicio SOLO por su tipo, horario, canales y distrito, sin decir para que tema o funcion sirve.")
    return f"MENSAJE DEL ESTUDIANTE:\n{user_text}\n\nRECURSOS (unica fuente permitida):\n{res}{extra}"

def llm_generate(user_text, resources, backend=None, feedback=None):
    """backend: 'openai_compat' (Ollama/vLLM/Latam-GPT servido via API compatible) o 'mock' (sin red, solo para probar el flujo)."""
    backend = backend or os.environ.get("AURA_LLM_BACKEND", "mock")
    if backend == "openai_compat":
        url = os.environ["AURA_LLM_URL"]            # p.ej. http://localhost:11434/v1/chat/completions
        body = {"model": os.environ["AURA_LLM_MODEL"], "temperature": 0.2,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": build_prompt(user_text, resources, feedback)}]}
        if os.environ.get("AURA_LLM_REASONING"): body["reasoning_effort"] = os.environ["AURA_LLM_REASONING"]   # opcional: p.ej. "low"
        global CACHE_HIT
        CACHE_HIT = False
        cache_path = os.environ.get("AURA_LLM_CACHE")   # opcional: guarda respuestas reales de Gemini para no gastar cuota al repetir
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        cache = {}
        if cache_path and os.path.exists(cache_path):
            try: cache = json.load(open(cache_path, encoding="utf-8"))
            except Exception: cache = {}
            if key in cache: CACHE_HIT = True; return cache[key]
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json",
                                     "Authorization": "Bearer " + os.environ.get("AURA_LLM_KEY", "none")})
        for intento in range(5):
            try:
                txt = json.load(urllib.request.urlopen(req, timeout=90))["choices"][0]["message"]["content"]
                if cache_path:
                    cache[key] = txt; json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
                return txt
            except urllib.error.HTTPError as e:
                detalle = e.read().decode("utf-8", "ignore")
                if e.code == 429 and "PerDay" in detalle:
                    raise RuntimeError("Cuota DIARIA de Gemini agotada (429): esperar al reinicio de cuota, usar otra clave o cambiar de modelo.") from e
                if e.code in (429, 500, 502, 503, 504) and intento < 4:
                    m = re.search(r"retry in ([\d.]+)s", detalle) or re.search(r'"retryDelay":\s*"([\d.]+)s"', detalle)
                    time.sleep(min(90, float(m.group(1)) + 2) if m else (5, 10, 20, 30)[intento]); continue
                raise RuntimeError(f"HTTP {e.code} de Gemini: {detalle[:200]}") from e
            except (urllib.error.URLError, TimeoutError):      # lectura lenta o corte de red: reintentar
                if intento < 4: time.sleep((3, 5, 10, 15)[intento]); continue
                raise
    # MOCK: NO es IA generativa. Plantilla para validar el flujo; en la demo real se usa un LLM.
    top = resources[0]
    return (f"Hola, soy una IA de orientacion (no reemplazo a un profesional). Por lo que cuentas, podria servirte "
            f"{top['name']} ({top['service_id']}): {top['why'][0]}; atiende {top['schedule']} por {', '.join(top['channels'])}. "
            "Tu decides si solicitarlo, y puedes hablar con una persona del equipo de bienestar cuando quieras.")

# ---------- 5. Validador ----------
DIAG_WORDS = ["diagnostic", "padeces", "tienes depresion", "tienes ansiedad", "trastorno", "medicacion", "farmaco", "tratamiento"]
def validate(answer, services):
    answer_n = re.sub(r"no (diagnostica|reemplaza|sustituye)\w*", "", norm(answer))
    ids = set(re.findall(r"SRV_AE_\d{3}", answer))
    problems = []
    if not ids: problems.append("no cita ningun service_id de D6")
    problems += [f"service_id inexistente: {i}" for i in ids if i not in services]
    problems += [f"lenguaje clinico: '{w}'" for w in DIAG_WORDS if w in answer_n]
    # En una linea que cita un servicio solo puede haber tipo, horario, canales, distrito y coincidencia con lo pedido.
    # D6 no declara temas ni funciones, asi que cualquier lexico tematico ahi es una atribucion inventada.
    TEMAS = ("presion", "estres", "ansiedad", "sobrecarga", "carga academica", "exigencia", "sueno", "empleab", "aislamiento",
             "navegacion", "navegar", "orientarte", "guiarte", "guiar", "sobrellevar", "afrontar")
    for linea in answer.splitlines():
        ln = norm(linea)
        if re.search(r"srv_ae_\d{3}", ln) and any(t in ln for t in TEMAS):
            problems.append("atribuye un tema/funcion al servicio (D6 no lo declara)"); break
    if not re.search(r"\bia\b", norm(answer)) and "inteligencia artificial" not in norm(answer): problems.append("no declara que es IA")
    return problems

# ---------- Orquestador ----------
def respond(user_text, services, country="Chile", district=None, backend=None):
    if detect_crisis(user_text):
        return {"route": "CRISIS", "answer": crisis_response(country), "resources": [], "problems": [], "need": None, "llm": "ninguno (regla de crisis)", "llm_error": None}
    need = classify_need_ml(user_text)
    res = retrieve(services, need, extract_prefs(user_text), district)
    used, err = backend or os.environ.get("AURA_LLM_BACKEND", "mock"), None
    try:
        ans = llm_generate(user_text, res, backend)
        if CACHE_HIT: used += " (respuesta guardada de una ejecución anterior)"
    except Exception as e:                       # sin red / cuota / clave invalida
        ans, used, err = llm_generate(user_text, res, "mock"), "mock", f"{type(e).__name__}: {e}"
    probs = validate(ans, services)
    for _ in range(2):                           # hasta 2 correcciones: se le indica al LLM que regla incumplio
        if not probs or used.startswith("mock"): break
        try:
            ans2 = llm_generate(user_text, res, backend, feedback=probs)
            probs2 = validate(ans2, services)
            ans, probs = ans2, probs2
            if not probs2: used = used.split(" (")[0] + " (corregido tras validador)"
        except Exception: break
    if probs:  # fallback seguro: nunca se muestra una salida que falle el validador
        ans, used = llm_generate(user_text, res, "mock"), "mock (validador bloqueo al LLM)"; probs = validate(ans, services)
    return {"route": "ORIENTACION", "answer": ans, "resources": res, "problems": probs, "need": need, "llm": used, "llm_error": err}
