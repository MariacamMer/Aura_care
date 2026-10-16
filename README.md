# AURA Care — Entregable 2 (Desafío Ciudad AETHERA)

**Equipo:** Grupo 9 — Bryan Silva Velasquez, Maria Camila Mercado, Daniel Jesus Huaraca Reyes (universidad / carrera: *completar*)
**Misiones:** M1 (estrés y ansiedad), M5 (estigma / baja búsqueda de ayuda), M6 (barreras de acceso)

## Qué hace
Agente de IA que orienta de forma preventiva y **no clínica** a estudiantes: entiende su necesidad, recupera servicios reales del mapa D6 y explica por qué los sugiere. Si detecta una crisis, suspende la recomendación automática y deriva a apoyo humano.

## Cómo funciona
`Mensaje → detector de crisis (reglas) → necesidad → RAG sobre D6 → LLM redacta → validador (sin servicios inventados ni lenguaje clínico) → respuesta`

## Modelos y herramientas de IA (COMPLETAR antes de entregar)
- LLM generativo: Gemini 3.8 Flash (`gemini-3.8-flash`), Google, vía API de Google AI Studio en nivel gratuito. Redacta la respuesta final usando solo los servicios recuperados de D6. *(confirmar nombre exacto y fecha de prueba antes de entregar)*
- Clasificador de necesidad: TF-IDF + regresión logística (scikit-learn) entrenado con D5
- Técnica: RAG con filtrado estructurado sobre D6 + reglas de seguridad deterministas

## Nota de reproducibilidad
Las llamadas a Gemini pueden guardarse en `cache_llm.json` (variable `AURA_LLM_CACHE`) para repetir casos sin gastar la cuota gratuita. Esas respuestas fueron generadas por el modelo; no se reescriben a mano. Si el modelo no responde, el agente lo indica en pantalla y usa una plantilla segura que no es IA generativa.

## Datasets del Data Pack utilizados
D6 (RAG de servicios), D5 (estructura de conversación y clasificador), D4 (análisis agregado de temas/tono), D1–D3 (score y métricas, *pendiente*). No se redistribuyen datos.

## Estructura
`src/aura_care.py` · `notebook/AURA_Care_demo.ipynb` · `docs/` (documento y ficha) · `FICHA_IA_RESPONSABLE.pdf`

## Video demostrativo
*Enlace (YouTube no listado / Drive)*
