
import os
import re
import json
import base64
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from groq import Groq

# ============================================================
# CORE SCIENCE AI - COMPLETE FLASK BACKEND
#
# Compatible with the existing frontend request format:
# prompt, task, subject, level, language, image, history
#
# Features:
# - Existing Biology, Chemistry and Physics teaching engines
# - Task detection and routing
# - Form/chapter routing using curriculum.json
# - Local knowledge retrieval from knowledge/ (.txt, .md, .json)
# - Image question support (model must support vision)
# - Conversation history
# - /health, /curriculum and /knowledge/status endpoints
#
# Important:
# This is a retrieval-enabled tutor backend, not a trained KSSM model.
# Add verified learning materials to knowledge/ for source-grounded answers.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
CURRICULUM_FILE = BASE_DIR / "curriculum.json"

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Set it as an environment variable before starting the app."
    )

client = Groq(api_key=GROQ_API_KEY)

# Configure these in your environment if your Groq account uses other model IDs.
GROQ_TEXT_MODEL = os.environ.get("GROQ_TEXT_MODEL", "openai/gpt-oss-120b")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")

# ============================================================
# CORE TEACHING INSTRUCTIONS
# ============================================================

CORE_IDENTITY = """
You are Core Science AI, a specialised Malaysian KSSM upper-secondary
Science tutor for Biology, Chemistry and Physics.

Help Form 4 and Form 5 students understand concepts, practise examination
questions, solve calculations, analyse diagrams, understand experiments,
and prepare revision material.

Be scientifically accurate, clear, structured, exam-aware, student-friendly,
and honest about uncertainty.

Do not claim that information is from an official KPM textbook unless the
actual source content has been supplied to the application.

Never fabricate chapter numbers, page numbers, marking schemes, experiment
results, textbook quotations, or official KPM wording.
"""

EXAM_ENGINE = """
KSSM / SPM ANSWERING ENGINE

For examination questions:
1. Identify what the question asks.
2. Identify the command word.
3. Identify the marks, if supplied.
4. Identify the relevant scientific concept.
5. Answer with enough distinct assessable points for the marks.
6. Use precise scientific terminology.
7. Avoid unrelated information.

Command words:
- STATE / NYATAKAN: give a direct fact, term, value or statement.
- DEFINE / TAKRIFKAN: give a precise definition.
- IDENTIFY / KENAL PASTI: name the required item.
- DESCRIBE / HURAIKAN: describe characteristics, sequence, structure or observations.
- EXPLAIN / JELASKAN: state the idea and explain the scientific reason or process.
- COMPARE / BANDINGKAN: show clear paired similarities or differences.
- CALCULATE / HITUNGKAN: show formula, substitution, working and unit.
- PREDICT / RAMALKAN: predict using the scientific relationship or evidence.
- SUGGEST / CADANGKAN: give a scientifically reasonable suggestion.
- JUSTIFY / WAJARKAN: give a claim and supporting scientific reason/evidence.

Do not assume one mark always equals one sentence. Focus on distinct
credit-worthy scientific ideas.
"""

TEACHING_ENGINE = """
TEACHING ENGINE

When explaining a difficult concept:
1. Start with a simple explanation.
2. Give the correct scientific explanation.
3. Break processes into logical steps.
4. Give a relevant example or analogy where useful.
5. Connect the concept to examination questions.
6. Mention common misconceptions where useful.

If the student says they do not understand, simplify the explanation and
break it into smaller steps. Keep correct scientific terminology.
"""

BIOLOGY_ENGINE = """
BIOLOGY ENGINE

Prioritise accurate biological terminology, structures and functions,
processes and sequences, cause-and-effect relationships, transport,
physiology, cell biology, genetics, ecology, reproduction, homeostasis,
coordination and experimental reasoning.

For a biological process, explain the starting condition, process,
mechanism, result and importance where relevant.

For a structure, connect structure, characteristic and function.
Do not invent biological facts to make an answer longer.
"""

CHEMISTRY_ENGINE = """
CHEMISTRY ENGINE

Prioritise chemical terminology, formulae, balanced equations, ionic
equations where appropriate, observations, inference, chemical reasoning,
mole calculations, concentration, acids and bases, periodic trends,
bonding, reaction rates, redox, carbon compounds and experiments.

For calculations show Given, Required, Formula, Substitution, Working,
Answer and Unit.

Check formula correctness, balancing, units and whether the result is
chemically reasonable.

For experiments distinguish Observation, Inference and Conclusion.
"""

PHYSICS_ENGINE = """
PHYSICS ENGINE

Prioritise physical quantities, definitions, equations, units, relationships
between variables, graphs, forces, motion, energy, heat, waves, light,
electricity, electromagnetism, pressure, nuclear physics and quantum physics.

For numerical questions show Given, Required, Formula, Substitution,
Working, Answer and Unit.

Check the formula before calculating. Show unit conversions where needed.

For graphs identify axes, gradient/intercept when relevant, and the
physical meaning of the graph.
"""

SUBJECT_ENGINES = {
    "biology": BIOLOGY_ENGINE,
    "biologi": BIOLOGY_ENGINE,
    "chemistry": CHEMISTRY_ENGINE,
    "kimia": CHEMISTRY_ENGINE,
    "physics": PHYSICS_ENGINE,
    "fizik": PHYSICS_ENGINE,
}

TASK_INSTRUCTIONS = {
    "explain": """
Explain the concept clearly and progressively. When useful use:
## Simple idea
## Scientific explanation
## Example
## Exam connection
## Common mistake
Do not turn a simple question into an unnecessarily long lecture.
""",
    "create_notes": """
Create detailed, useful revision notes. Include key ideas, definitions,
concepts, processes or mechanisms, equations where relevant, examples,
common exam mistakes, SPM keywords and a recap. Do not reduce the notes
to one-line facts.
""",
    "generate_questions": """
Generate original KSSM/SPM-style practice questions suitable for the
selected level. Mix objective, structured, calculation, application,
HOTS and experiment questions when relevant. Include marks. Do not claim
that generated questions are actual SPM questions.
""",
    "solve": """
Solve step by step. For calculations show Given, Required, Formula,
Substitution, Working, Final answer and Unit. Explain why the formula
or method is appropriate.
""",
    "generate_marking_scheme": """
Create a marking guide with distinct credit-worthy points, appropriate
scientific keywords, acceptable alternatives where justified, and common
mistakes. Do not claim it is an official LPM marking scheme.
""",
    "mark_answer": """
Act as a fair examiner and tutor. Explain what the question required,
what the student wrote, correct ideas, missing ideas, incorrect ideas,
likely credit where enough information is available, and how to improve.
Do not invent an official mark allocation.
""",
    "summarize": """
Create a concise but useful summary. Include core concepts, definitions,
formulae, processes, exam keywords and common mistakes. Preserve essential
scientific relationships.
""",
    "quiz": """
Run an interactive quiz. Ask one question at a time and wait for the
student's response. Do not reveal the answer before the student attempts it.
""",
}

TASK_KEYWORDS = {
    "solve": [
        "calculate", "calculation", "solve", "find the value", "determine",
        "hitung", "hitungkan", "work out"
    ],
    "create_notes": [
        "notes", "nota", "revision notes", "study notes", "make notes",
        "buat nota"
    ],
    "generate_questions": [
        "generate questions", "practice questions", "exam questions",
        "trial questions", "soalan", "create questions"
    ],
    "generate_marking_scheme": [
        "marking scheme", "mark scheme", "skema jawapan"
    ],
    "mark_answer": [
        "mark my answer", "check my answer", "is my answer correct",
        "how many marks", "check this answer", "mark this"
    ],
    "summarize": ["summarize", "summary", "ringkaskan", "ringkasan"],
    "quiz": ["quiz me", "test me", "ask me one question"],
    "explain": ["explain", "what is", "why", "how does", "how do", "jelaskan", "terangkan"],
}

# ============================================================
# CURRICULUM DATABASE
# The directory is a routing aid. Verify it against current DSKP and
# textbook editions before treating it as an official source.
# ============================================================

DEFAULT_CURRICULUM = {
    "Biology": {
        "Form 4": [
            {"chapter": 1, "title": "Introduction to Biology and Laboratory Rules"},
            {"chapter": 2, "title": "Cell Biology and Organisation"},
            {"chapter": 3, "title": "Movement of Substances Across the Plasma Membrane"},
            {"chapter": 4, "title": "Chemical Composition in a Cell"},
            {"chapter": 5, "title": "Metabolism and Enzymes"},
            {"chapter": 6, "title": "Cell Division"},
            {"chapter": 7, "title": "Cellular Respiration"},
            {"chapter": 8, "title": "Respiratory System in Humans and Animals"},
            {"chapter": 9, "title": "Nutrition and Human Digestive System"},
            {"chapter": 10, "title": "Transport in Humans and Animals"},
            {"chapter": 11, "title": "Immunity in Humans"},
            {"chapter": 12, "title": "Coordination and Response in Humans"},
            {"chapter": 13, "title": "Homeostasis and Human Urinary System"},
            {"chapter": 14, "title": "Support and Movement in Humans and Animals"},
            {"chapter": 15, "title": "Sexual Reproduction, Development and Growth"}
        ],
        "Form 5": [
            {"chapter": 1, "title": "Organisation of Plant Tissues and Growth"},
            {"chapter": 2, "title": "Leaf Structure and Function"},
            {"chapter": 3, "title": "Nutrition in Plants"},
            {"chapter": 4, "title": "Transport in Plants"},
            {"chapter": 5, "title": "Response in Plants"},
            {"chapter": 6, "title": "Sexual Reproduction in Flowering Plants"},
            {"chapter": 7, "title": "Adaptations of Plants in Different Habitats"},
            {"chapter": 8, "title": "Biodiversity"},
            {"chapter": 9, "title": "Ecosystem"},
            {"chapter": 10, "title": "Environmental Sustainability"},
            {"chapter": 11, "title": "Inheritance"},
            {"chapter": 12, "title": "Variation"},
            {"chapter": 13, "title": "Genetic Technology"}
        ]
    },
    "Chemistry": {
        "Form 4": [
            {"chapter": 1, "title": "Introduction to Chemistry"},
            {"chapter": 2, "title": "Matter and the Atomic Structure"},
            {"chapter": 3, "title": "The Mole Concept, Chemical Formula and Equation"},
            {"chapter": 4, "title": "The Periodic Table of Elements"},
            {"chapter": 5, "title": "Chemical Bond"},
            {"chapter": 6, "title": "Acids, Bases and Salts"},
            {"chapter": 7, "title": "Rate of Reaction"},
            {"chapter": 8, "title": "Manufactured Substances in Industry"}
        ],
        "Form 5": [
            {"chapter": 1, "title": "Redox Equilibrium"},
            {"chapter": 2, "title": "Carbon Compounds"},
            {"chapter": 3, "title": "Thermochemistry"},
            {"chapter": 4, "title": "Polymer"},
            {"chapter": 5, "title": "Consumer and Industrial Chemistry"}
        ]
    },
    "Physics": {
        "Form 4": [
            {"chapter": 1, "title": "Measurement"},
            {"chapter": 2, "title": "Force and Motion"},
            {"chapter": 3, "title": "Gravitation"},
            {"chapter": 4, "title": "Heat"},
            {"chapter": 5, "title": "Waves"},
            {"chapter": 6, "title": "Light and Optics"}
        ],
        "Form 5": [
            {"chapter": 1, "title": "Force and Motion II"},
            {"chapter": 2, "title": "Pressure"},
            {"chapter": 3, "title": "Electricity"},
            {"chapter": 4, "title": "Electromagnetism"},
            {"chapter": 5, "title": "Electronics"},
            {"chapter": 6, "title": "Nuclear Physics"},
            {"chapter": 7, "title": "Quantum Physics"}
        ]
    }
}

def load_curriculum():
    try:
        with CURRICULUM_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
        curriculum = data.get("curriculum", data)
        if isinstance(curriculum, dict) and curriculum:
            return curriculum
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Curriculum file error; using built-in directory: {exc}")
    return DEFAULT_CURRICULUM

CURRICULUM = load_curriculum()

def normalise_subject(value: Any) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "bio": "Biology", "biology": "Biology", "biologi": "Biology",
        "chem": "Chemistry", "chemistry": "Chemistry", "kimia": "Chemistry",
        "phy": "Physics", "physics": "Physics", "fizik": "Physics",
    }
    return aliases.get(raw, str(value or "General").strip().title())

def detect_form(text: str, level: str = "") -> str:
    # Prefer an explicit form in the student's question.
    prompt_text = (text or "").lower()
    match = re.search(r"\b(?:form|tingkatan)\s*([45])\b|\bf([45])\b", prompt_text)
    if match:
        return match.group(1) or match.group(2)

    # Only use the selected level when it identifies ONE form.
    # A broad value such as "Form 4-5 / SPM" must not silently become Form 4.
    level_text = (level or "").lower().strip()
    exact_match = re.fullmatch(r"(?:form|tingkatan)\s*([45])", level_text)
    if exact_match:
        return exact_match.group(1)

    return ""

def get_chapters(subject: str, form: str):
    if not form:
        return []
    return CURRICULUM.get(subject, {}).get(f"Form {form}", [])

def detect_chapter(prompt: str, subject: str, form: str):
    chapters = get_chapters(subject, form)
    text = (prompt or "").lower()
    match = re.search(r"\b(?:chapter|chap|ch|bab)\s*\.?\s*(\d{1,2})\b", text)
    if match:
        number = int(match.group(1))
        for item in chapters:
            if int(item.get("chapter", -1)) == number:
                return item

    for item in chapters:
        title = str(item.get("title", "")).lower()
        if title and title in text:
            return item
    return None

def get_curriculum_context(subject: str, prompt: str, level: str):
    form = detect_form(prompt, level)
    chapters = get_chapters(subject, form)
    detected = detect_chapter(prompt, subject, form)
    lines = "\n".join(
        f"- Chapter {x.get('chapter')}: {x.get('title')}" for x in chapters
    )
    return {
        "form": form or None,
        "chapter": detected,
        "context": (
            f"Subject: {subject}\nForm: {form or 'not identified'}\n"
            f"Chapter directory:\n{lines or 'No form-specific chapter list available.'}\n"
            "This directory is a routing aid, not proof of exact DSKP learning standards. "
            "Do not invent subtopics, Content Standards, Learning Standards or page numbers."
        )
    }

# ============================================================
# KNOWLEDGE FOLDER + LIGHTWEIGHT RETRIEVAL
# Supported: .txt, .md, .json
# ============================================================

SUPPORTED_KNOWLEDGE_EXTENSIONS = {".txt", ".md", ".json"}
KNOWLEDGE_INDEX = []

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_into_chunks(text: str, source: str, chunk_size: int = 1400, overlap: int = 200):
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + int(chunk_size * 0.6), end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"source": source, "text": chunk})
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

def read_knowledge_file(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            try:
                content = json.dumps(json.loads(content), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pass
        return content
    except (OSError, UnicodeDecodeError) as exc:
        print(f"Skipping knowledge file {path.name}: {exc}")
        return ""

def rebuild_knowledge_index():
    global KNOWLEDGE_INDEX
    KNOWLEDGE_INDEX = []
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    for path in KNOWLEDGE_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_KNOWLEDGE_EXTENSIONS:
            relative = str(path.relative_to(KNOWLEDGE_DIR))
            KNOWLEDGE_INDEX.extend(
                split_into_chunks(read_knowledge_file(path), relative)
            )
    print(f"Knowledge index ready: {len(KNOWLEDGE_INDEX)} chunks")

def token_set(text: str):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

def search_knowledge(prompt: str, subject: str, form: str = "", limit: int = 5):
    query_tokens = token_set(prompt)
    if not query_tokens:
        return []

    scored = []
    for chunk in KNOWLEDGE_INDEX:
        body = chunk["text"]
        body_tokens = token_set(body)
        overlap = len(query_tokens & body_tokens)
        if overlap == 0:
            continue

        score = overlap / max(len(query_tokens), 1)
        metadata = f"{chunk['source']} {body[:300]}".lower()

        if subject.lower() in metadata:
            score += 0.15
        if form and (f"form {form}" in metadata or f"tingkatan {form}" in metadata):
            score += 0.10

        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for score, chunk in scored[:limit] if score > 0]

def format_retrieved_sources(chunks):
    if not chunks:
        return (
            "No matching local knowledge passages were found for this question. "
            "Do not claim that the answer came from a local textbook file."
        )
    output = []
    for index, chunk in enumerate(chunks, 1):
        output.append(
            f"[LOCAL SOURCE {index}: {chunk['source']}]\n{chunk['text']}"
        )
    return "\n\n".join(output)

# ============================================================
# ROUTING + IMAGE HELPERS
# ============================================================

def detect_task(text: str, supplied_task: str = "") -> str:
    allowed = set(TASK_INSTRUCTIONS)
    supplied = (supplied_task or "").strip().lower()
    if supplied in allowed:
        return supplied

    lower = (text or "").lower()
    for task, keywords in TASK_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            return task
    return "explain"

def looks_like_calculation(text: str) -> bool:
    lower = (text or "").lower()
    patterns = [
        r"\bcalculate\b", r"\bwork out\b", r"\bhitungkan\b",
        r"\bfind the value\b", r"\bdetermine\b",
        r"\d+(?:\.\d+)?\s*(?:m/s|km/h|cm|mm|km|kg|mol|mol\s*dm-3|v|a|w|j|pa|hz|°c)\b",
        r"\b(?:mass|volume|concentration|velocity|speed|acceleration|force|energy|power|current|voltage|resistance|frequency|wavelength|moles)\b"
    ]
    return any(re.search(pattern, lower) for pattern in patterns)

def validate_image_data(image_value):
    if not isinstance(image_value, str) or not image_value.strip():
        return None
    value = image_value.strip()
    if not value.startswith("data:image/") or ";base64," not in value:
        return None
    header, encoded = value.split(",", 1)
    if not re.match(r"^data:image/(png|jpeg|jpg|webp|gif);base64$", header, re.I):
        return None
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error):
        return None
    if len(decoded) > 8 * 1024 * 1024:
        return None
    return value

def image_instruction(subject: str):
    focus = {
        "Biology": "Focus on biological structures, labels, processes, graphs and experimental diagrams.",
        "Chemistry": "Focus on apparatus, equations, tables, graphs, observations, structures and experiment questions.",
        "Physics": "Focus on diagrams, circuits, graphs, rays, forces, motion and experimental setups."
    }.get(subject, "Analyse the visible educational content carefully.")
    return (
        "IMAGE ANALYSIS MODE\n" + focus + "\n"
        "Do not invent unreadable values or labels. State when any part is unclear. "
        "Separate what is visibly shown from scientific inference."
    )

def clean_history(history):
    if not isinstance(history, list):
        return []
    cleaned = []
    for item in history[-8:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            cleaned.append({"role": role, "content": content[:8000]})
    return cleaned

def build_system_prompt(subject, level, language, task, prompt, curriculum_info, sources, has_image):
    engine = SUBJECT_ENGINES.get(subject.lower(), "")
    chapter = curriculum_info.get("chapter")
    if chapter:
        chapter_text = (
            f"Detected curriculum location: Chapter {chapter.get('chapter')} - "
            f"{chapter.get('title')}. Treat this as a routing clue only."
        )
    else:
        chapter_text = "No exact chapter was confidently detected. Do not invent one."

    retrieved = format_retrieved_sources(sources)
    image_text = image_instruction(subject) if has_image else ""

    return f"""
{CORE_IDENTITY}

{EXAM_ENGINE}

{TEACHING_ENGINE}

{engine}

TASK MODE:
{TASK_INSTRUCTIONS.get(task, TASK_INSTRUCTIONS['explain'])}

CURRENT STUDENT CONTEXT:
Subject: {subject}
Level: {level}
Requested language: {language}
Task: {task}
Question: {prompt}

CURRICULUM ROUTING:
{curriculum_info.get('context', '')}
{chapter_text}

{image_text}

RETRIEVED LOCAL KNOWLEDGE:
{retrieved}

SOURCE AND ACCURACY RULES:
- Use relevant retrieved passages as the primary reference for source-specific details.
- The source snippets may be incomplete. Do not infer missing textbook content.
- If no relevant local source was retrieved, use general scientific knowledge carefully.
- Never claim an answer is directly from an official textbook unless that source was supplied.
- Never fabricate page numbers, quotations, official learning standards or marking schemes.
- If the source does not support a detail, say so where relevant.
- Answer in {language or 'English'} unless the student clearly asks for another language.
- Be detailed enough to teach, but do not add irrelevant material.
""".strip()

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        response = send_from_directory(str(BASE_DIR), "index.html")
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response
    return jsonify({
        "success": True,
        "message": "Core Science AI backend is running. Add index.html to this folder."
    })

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            return jsonify({"success": False, "error": "Please enter a question."}), 400

        subject = normalise_subject(data.get("subject", "Biology"))
        level = str(data.get("level", "Form 4-5 / SPM")).strip()
        language = str(data.get("language", "English")).strip() or "English"
        frontend_task = str(data.get("task", "")).strip()
        history = clean_history(data.get("history", []))

        image_data = validate_image_data(data.get("image"))
        has_image = bool(image_data)

        task = detect_task(prompt, frontend_task)
        if looks_like_calculation(prompt):
            task = "solve"

        curriculum_info = get_curriculum_context(subject, prompt, level)
        form = curriculum_info.get("form") or ""
        retrieved_chunks = search_knowledge(prompt, subject, form, limit=5)

        system_prompt = build_system_prompt(
            subject=subject,
            level=level,
            language=language,
            task=task,
            prompt=prompt,
            curriculum_info=curriculum_info,
            sources=retrieved_chunks,
            has_image=has_image
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)

        if has_image:
            user_content = [
                {"type": "text", "text": prompt or "Please analyse this educational image."},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]
        else:
            user_content = prompt

        messages.append({"role": "user", "content": user_content})

        model = GROQ_VISION_MODEL if has_image else GROQ_TEXT_MODEL

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=4096
        )

        reply = completion.choices[0].message.content or ""
        chapter = curriculum_info.get("chapter")

        return jsonify({
            "success": True,
            "reply": reply,
            "task": task,
            "subject": subject,
            "level": level,
            "model": model,
            "routing": {
                "form": form or None,
                "chapter": chapter,
                "retrieved_source_count": len(retrieved_chunks),
                "retrieved_sources": [x["source"] for x in retrieved_chunks]
            }
        })

    except Exception as exc:
        print("CHAT ERROR:", repr(exc))
        # Keep the detailed error in the server terminal, not exposed to visitors.
        return jsonify({
            "success": False,
            "error": "The AI request failed. Check the Flask terminal for details. "
                     "Confirm your API key and model names."
        }), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "Core Science AI",
        "text_model": GROQ_TEXT_MODEL,
        "vision_model": GROQ_VISION_MODEL,
        "subjects": ["Biology", "Chemistry", "Physics"],
        "knowledge_chunks": len(KNOWLEDGE_INDEX)
    })

@app.route("/curriculum", methods=["GET"])
def curriculum_route():
    return jsonify({"success": True, "curriculum": CURRICULUM})

@app.route("/knowledge/status", methods=["GET"])
def knowledge_status():
    files = []
    if KNOWLEDGE_DIR.exists():
        for path in KNOWLEDGE_DIR.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_KNOWLEDGE_EXTENSIONS:
                files.append(str(path.relative_to(KNOWLEDGE_DIR)))
    return jsonify({
        "success": True,
        "folder": "knowledge",
        "supported_extensions": sorted(SUPPORTED_KNOWLEDGE_EXTENSIONS),
        "file_count": len(files),
        "chunk_count": len(KNOWLEDGE_INDEX),
        "files": files
    })

@app.route("/knowledge/reload", methods=["POST"])
def knowledge_reload():
    rebuild_knowledge_index()
    return jsonify({
        "success": True,
        "message": "Knowledge index reloaded.",
        "chunk_count": len(KNOWLEDGE_INDEX)
    })

# Build the local search index at startup.
rebuild_knowledge_index()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
