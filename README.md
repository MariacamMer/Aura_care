# AURA Care — Entregable 2 (Desafío Ciudad AETHERA)

**Equipo:** Grupo 9
- Bryan Silva Velasquez — Universidad del Bío-Bío, Ingeniería Civil en Informática
- María Camila Mercado — Universidad de Atacama, Ingeniería en Alimentos
- Daniel Jesús Huaraca Reyes — Universidad Peruana de Ciencias Aplicadas, Ingeniería Mecatrónica

**Misiones:** M1 (estrés y ansiedad), M5 (estigma / baja búsqueda de ayuda), M6 (barreras de acceso)

## Qué hace
Agente de IA que orienta de forma preventiva y **no clínica** a estudiantes: entiende su necesidad, recupera servicios reales del mapa D6 y explica por qué los sugiere. Si detecta una crisis, suspende la recomendación automática y deriva a apoyo humano.

**Lo que no hace:** no diagnostica, no interpreta síntomas, no sugiere tratamientos, no simula terapia y no decide por la persona. Apoya, no reemplaza, la atención profesional.

## Cómo funciona
`Mensaje → detector de crisis (reglas) → necesidad → RAG sobre D6 → LLM redacta → validador (sin servicios inventados ni lenguaje clínico) → respuesta`

| Capa | Qué hace | Técnica |
|---|---|---|
| 1. Seguridad de crisis | Corre antes que todo; si se activa, el LLM no interviene y se entregan líneas de ayuda (Chile y Perú) | Léxico propio de 18 patrones |
| 2. Necesidad | Clasifica el motivo: presión académica, sueño y rutina, apoyo social, preocupación laboral o navegación de servicios | Reglas léxicas; si no coinciden, TF-IDF + regresión logística entrenada con D5 |
| 3. Recuperación (RAG) | Selecciona servicios de D6 según tipo, canal pedido, horario extendido y distrito, y registra el motivo de cada selección | Filtrado estructurado sobre D6 |
| 4. Generación | Redacta una respuesta breve usando solo los servicios recuperados | Gemini 3.8 Flash |
| 5. Validación | Rechaza `service_id` inexistentes, lenguaje clínico, temas atribuidos a un servicio que D6 no declara, o respuestas que no declaren que son IA. Pide hasta 2 correcciones al modelo; si persiste, usa una plantilla segura | Reglas sobre la salida |

El LLM **no elige** los servicios (lo hace la recuperación), **no decide** si hay crisis (lo hace la regla) y **no puede introducir** servicios nuevos (lo bloquea el validador).

## Modelos y herramientas de IA
- **LLM generativo:** Gemini 3.8 Flash (`gemini-3.8-flash`), Google, vía API de Google AI Studio en nivel gratuito (endpoint compatible con OpenAI). Redacta la respuesta final usando solo los servicios recuperados de D6. Probado en octubre de 2026.
- **Clasificador de necesidad:** TF-IDF + regresión logística (scikit-learn), entrenado con D5. Se usa solo como respaldo cuando las reglas léxicas no coinciden.
- **Técnica:** RAG con filtrado estructurado sobre D6 + reglas de seguridad deterministas.
- **Entorno de ejecución:** Google Colab.

## Nota de reproducibilidad
Las llamadas a Gemini pueden guardarse en `cache_llm.json` (variable `AURA_LLM_CACHE`) para repetir casos sin gastar la cuota gratuita. Esas respuestas fueron generadas por el modelo; no se reescriben a mano. El archivo se crea al ejecutar el notebook con clave de Gemini.

Si el modelo no responde (sin red, cuota agotada o clave inválida), el agente lo indica en pantalla (`llm_error`) y usa una plantilla segura que **no es IA generativa** (backend `mock`).

## Cómo ejecutarlo

**1. Datos.** El Data Pack no se redistribuye. Coloca los archivos en una carpeta `datos/` junto a `src/` (o define `AURA_DATA_DIR`):
`D1_wellbeing_survey.csv`, `D2_support_services.csv`, `D3_academic_trajectory.csv`, `D4_testimonials.json`, `D5_help_line_conversations.json`, `D6_services_map.geojson`.

**2. Dependencias.**
```bash
pip install -r requirements.txt
```

**3. Clave de Gemini** (gratuita en [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)). No pegarla en el notebook ni subirla a GitHub.
```bash
export GEMINI_API_KEY="tu_clave"          # Windows PowerShell: $env:GEMINI_API_KEY="tu_clave"
```

**4. Prueba rápida (1 llamada):**
```bash
cd src
python probar_llm.py
```
Debe terminar con `>>> RESULTADO: Gemini funciona`.

**5. Demo completa:** abrir `notebook/AURA_Care_demo.ipynb` (en Colab, guardar la clave en *Secretos* como `GEMINI_API_KEY`). Sin clave, el notebook corre con `mock` y lo avisa.

## Datasets del Data Pack utilizados
| Dataset | Uso |
|---|---|
| D1 Encuesta de bienestar | Score preventivo agregado y reconstrucción de las líneas base M1, M5 y M6 |
| D2 Uso de servicios | Esperas, asistencia por canal, presión por distrito y brecha de acceso |
| D3 Trayectoria académica | Validación externa del score (asistencia y alertas académicas, que no entran al score) |
| D4 Testimonios | Análisis agregado de temas y tono; no se usa para clasificar |
| D5 Conversaciones de orientación | Estructura de la conversación y entrenamiento del clasificador de necesidad |
| D6 Mapa de servicios | Fuente única de servicios para el RAG |

D7 (calendario) no se utilizó; queda como mejora. No se redistribuyen datos.

## Resultados principales
- **Líneas base reproducidas** con D1 y D2: M1 41,0 % · M5 68,0 % · M6 44,0 % desconoce los servicios y 6,0 semanas de espera media.
- **Score preventivo** (agregado, no clínico, reglas del equipo): niveles bajo (0–3), medio (4–6) y alto (7+). Correlación con alertas académicas 0,47 y con asistencia −0,50 (variables de D3 que no entran al score).
- **Grupo prioritario** (nivel alto, 21,3 %): 75,0 % nunca consultó apoyo y 57,6 % desconoce los servicios.
- **Seguridad:** prueba de humo propia con 12 mensajes de crisis y 12 que no lo son: 12/12 detectados, 0 falsos positivos. Conjunto pequeño y no validado por profesionales; sirve para verificar el flujo, no para afirmar desempeño general.

El prototipo **no mide reducción** de ansiedad, de no-consulta ni de espera: eso exige un piloto con estudiantes reales y apoyo profesional.

## Limitaciones
- Datos 100 % sintéticos; clasificador y score deben revalidarse con texto y datos reales.
- D5 usa plantillas (17 textos distintos en 2.500 conversaciones): el clasificador puede generalizar mal a texto libre.
- El léxico de crisis es propio y pequeño; debe ampliarse y validarse con profesionales (incluyendo expresiones de Chile y Perú). Las líneas de ayuda deben verificarse antes de cualquier uso real.
- La supervisión humana descrita en la Ficha de IA Responsable es una propuesta de diseño, no está implementada.
- El nivel gratuito de Google puede usar los prompts para mejorar sus productos: **no ingresar datos personales reales**.

## Estructura
```
aura_care/
├── README.md
├── requirements.txt
├── FICHA_IA_RESPONSABLE.pdf
├── docs/
│   └── Entregable2_AURA_Care.pdf
├── notebook/
│   └── AURA_Care_demo.ipynb
└── src/
    ├── aura_care.py      # agente: crisis, necesidad, RAG, LLM, validador
    ├── preventive.py     # score preventivo agregado (D1)
    └── probar_llm.py     # prueba rápida de Gemini
```

## Video demostrativo
https://drive.google.com/file/d/1HGHPZsUPHV_cZJ97QDXSkNjATBSwxPmT/view?usp=sharing
