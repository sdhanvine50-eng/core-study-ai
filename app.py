
import os, re, json, base64
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from groq import Groq

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
CURRICULUM_FILE = BASE_DIR / "curriculum.json"

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing.")

client = Groq(api_key=API_KEY)
TEXT_MODEL = os.environ.get("GROQ_TEXT_MODEL", "openai/gpt-oss-120b")
VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")

CORE = """
You are Core Science AI, a specialised Malaysian KSSM Form 4 and Form 5
learning assistant for Biology, Chemistry and Physics.

Your job is to TEACH, not merely answer.

Priorities:
1. Scientific accuracy
2. KSSM relevance
3. Clear understanding
4. Examination usefulness
5. Correct terminology
6. Honest uncertainty

Never invent official KPM/LPM wording, DSKP standards, textbook pages,
textbook quotations, experiments, results or official marking schemes.

When local authorised source material is supplied, use it as the primary
source for source-specific information. If it is insufficient, say so.

Answer in the requested language.

If the student says they do not understand, simplify and teach from the
beginning. If they ask for notes, make structured useful notes. If they
ask for an exam answer, focus on credit-worthy scientific points.
"""

EXAM = """
EXAM ENGINE

STATE/NYATAKAN = direct fact, term, value or statement.
DEFINE/TAKRIFKAN = precise definition.
IDENTIFY/KENAL PASTI = name the item.
DESCRIBE/HURAIKAN = characteristics, observations, structures or sequence.
EXPLAIN/JELASKAN = idea + scientific reason/mechanism.
COMPARE/BANDINGKAN = clear paired similarities/differences.
CALCULATE/HITUNGKAN = Given, Required, Formula, Substitution, Working,
Answer and Unit.
PREDICT/RAMALKAN = prediction supported by evidence.
SUGGEST/CADANGKAN = scientifically reasonable suggestion.
JUSTIFY/WAJARKAN = conclusion + scientific reason/evidence.
EVALUATE/NILAI = evidence, factors/limitations and justified conclusion.

If marks are given, provide enough distinct credit-worthy scientific ideas.
Do not mechanically assume one mark equals one sentence.
"""

TEACHING = """
TEACHING ENGINE

For explanations, use the depth needed:
- Simple idea
- Scientific explanation
- Step-by-step process
- Example/analogy when useful
- Exam connection
- Common mistake when useful

Do not force every section into every answer.
Never let an analogy replace the real scientific explanation.
"""

BIOLOGY = """
BIOLOGY:
Use accurate biological terminology. Focus on structures, functions,
processes, cause-and-effect, transport, physiology, cells, genetics,
ecology, reproduction, homeostasis, coordination and experiments.
For processes explain condition, mechanism, result and importance where useful.
"""

CHEMISTRY = """
CHEMISTRY:
Use accurate formulae, equations, observations, inference and chemical
reasoning. Check balancing, units and calculations. For experiments separate
Observation, Inference and Conclusion.
"""

PHYSICS = """
PHYSICS:
Use accurate quantities, definitions, equations and units. Check formulas,
unit conversions and whether numerical answers are physically reasonable.
For graphs explain axes, gradient/intercept and physical meaning when useful.
"""

SUBJECT_ENGINE = {
    "biology": BIOLOGY,
    "chemistry": CHEMISTRY,
    "physics": PHYSICS
}

TASKS = {
    "explain": "Explain clearly from simple idea to scientific explanation.",
    "create_notes": "Create detailed revision notes with definitions, concepts, processes, equations, examples, exam keywords, mistakes and recap.",
    "generate_questions": "Generate original KSSM/SPM-style practice questions with marks. Do not claim they are actual SPM questions.",
    "solve": "Solve step by step using Given, Required, Formula, Substitution, Working, Answer and Unit.",
    "generate_marking_scheme": "Create a non-official marking guide with distinct credit-worthy points and scientific keywords.",
    "mark_answer": "Analyse the student's answer, identify correct/missing/incorrect ideas and explain how to improve. Do not invent official marks.",
    "summarize": "Create a concise useful summary containing core concepts, definitions, processes, equations and exam keywords.",
    "quiz": "Ask one question at a time and wait for the student's response."
}

KEYWORDS = {
    "solve": ["calculate","calculation","solve","find the value","determine",
              "hitung","hitungkan","work out"],
    "create_notes": ["notes","nota","revision notes","study notes","make notes","buat nota"],
    "generate_questions": ["generate questions","practice questions","exam questions",
                           "trial questions","soalan","create questions"],
    "generate_marking_scheme": ["marking scheme","mark scheme","skema jawapan"],
    "mark_answer": ["mark my answer","check my answer","is my answer correct",
                    "how many marks","check this answer","mark this"],
    "summarize": ["summarize","summary","ringkaskan","ringkasan"],
    "quiz": ["quiz me","test me","ask me one question"],
    "explain": ["explain","what is","why","how does","how do","jelaskan","terangkan"]
}

DEFAULT_CURRICULUM = {
    "Biology": {
        "Form 4": [
            {"chapter":1,"title":"Introduction to Biology and Laboratory Rules"},
            {"chapter":2,"title":"Cell Biology and Organisation"},
            {"chapter":3,"title":"Movement of Substances Across the Plasma Membrane"},
            {"chapter":4,"title":"Chemical Composition in a Cell"},
            {"chapter":5,"title":"Metabolism and Enzymes"},
            {"chapter":6,"title":"Cell Division"},
            {"chapter":7,"title":"Cellular Respiration"},
            {"chapter":8,"title":"Respiratory System in Humans and Animals"},
            {"chapter":9,"title":"Nutrition and Human Digestive System"},
            {"chapter":10,"title":"Transport in Humans and Animals"},
            {"chapter":11,"title":"Immunity in Humans"},
            {"chapter":12,"title":"Coordination and Response in Humans"},
            {"chapter":13,"title":"Homeostasis and Human Urinary System"},
            {"chapter":14,"title":"Support and Movement in Humans and Animals"},
            {"chapter":15,"title":"Sexual Reproduction, Development and Growth"}
        ],
        "Form 5": [
            {"chapter":1,"title":"Organisation of Plant Tissues and Growth"},
            {"chapter":2,"title":"Leaf Structure and Function"},
            {"chapter":3,"title":"Nutrition in Plants"},
            {"chapter":4,"title":"Transport in Plants"},
            {"chapter":5,"title":"Response in Plants"},
            {"chapter":6,"title":"Sexual Reproduction in Flowering Plants"},
            {"chapter":7,"title":"Adaptations of Plants in Different Habitats"},
            {"chapter":8,"title":"Biodiversity"},
            {"chapter":9,"title":"Ecosystem"},
            {"chapter":10,"title":"Environmental Sustainability"},
            {"chapter":11,"title":"Inheritance"},
            {"chapter":12,"title":"Variation"},
            {"chapter":13,"title":"Genetic Technology"}
        ]
    },
    "Chemistry": {
        "Form 4": [
            {"chapter":1,"title":"Introduction to Chemistry"},
            {"chapter":2,"title":"Matter and the Atomic Structure"},
            {"chapter":3,"title":"The Mole Concept, Chemical Formula and Equation"},
            {"chapter":4,"title":"The Periodic Table of Elements"},
            {"chapter":5,"title":"Chemical Bond"},
            {"chapter":6,"title":"Acids, Bases and Salts"},
            {"chapter":7,"title":"Rate of Reaction"},
            {"chapter":8,"title":"Manufactured Substances in Industry"}
        ],
        "Form 5": [
            {"chapter":1,"title":"Redox Equilibrium"},
            {"chapter":2,"title":"Carbon Compounds"},
            {"chapter":3,"title":"Thermochemistry"},
            {"chapter":4,"title":"Polymer"},
            {"chapter":5,"title":"Consumer and Industrial Chemistry"}
        ]
    },
    "Physics": {
        "Form 4": [
            {"chapter":1,"title":"Measurement"},
            {"chapter":2,"title":"Force and Motion"},
            {"chapter":3,"title":"Gravitation"},
            {"chapter":4,"title":"Heat"},
            {"chapter":5,"title":"Waves"},
            {"chapter":6,"title":"Light and Optics"}
        ],
        "Form 5": [
            {"chapter":1,"title":"Force and Motion II"},
            {"chapter":2,"title":"Pressure"},
            {"chapter":3,"title":"Electricity"},
            {"chapter":4,"title":"Electromagnetism"},
            {"chapter":5,"title":"Electronics"},
            {"chapter":6,"title":"Nuclear Physics"},
            {"chapter":7,"title":"Quantum Physics"}
        ]
    }
}

def load_curriculum():
    try:
        data = json.loads(CURRICULUM_FILE.read_text(encoding="utf-8"))
        return data.get("curriculum", data)
    except Exception:
        return DEFAULT_CURRICULUM

CURRICULUM = load_curriculum()

def normalise_subject(value):
    x = str(value or "").lower().strip()
    return {
        "bio":"Biology","biology":"Biology","biologi":"Biology",
        "chem":"Chemistry","chemistry":"Chemistry","kimia":"Chemistry",
        "phy":"Physics","physics":"Physics","fizik":"Physics"
    }.get(x, str(value or "General").title())

def detect_form(text, level=""):
    m = re.search(r"\b(?:form|tingkatan)\s*([45])\b|\bf([45])\b",
                  (text or "").lower())
    if m:
        return m.group(1) or m.group(2)

    m = re.fullmatch(r"(?:form|tingkatan)\s*([45])",
                     (level or "").lower().strip())
    return m.group(1) if m else ""

def chapters(subject, form):
    return CURRICULUM.get(subject, {}).get(f"Form {form}", []) if form else []

def detect_chapter(prompt, subject, form):
    text = (prompt or "").lower()
    items = chapters(subject, form)

    m = re.search(r"\b(?:chapter|chap|ch|bab)\s*(\d{1,2})\b", text)
    if m:
        n = int(m.group(1))
        return next((x for x in items if x["chapter"] == n), None)

    for x in items:
        if x["title"].lower() in text:
            return x

    return None

def curriculum_info(subject, prompt, level):
    form = detect_form(prompt, level)
    items = chapters(subject, form)
    detected = detect_chapter(prompt, subject, form)

    listing = "\n".join(
        f"- Chapter {x['chapter']}: {x['title']}" for x in items
    )

    return {
        "form": form,
        "chapter": detected,
        "context":
            f"Subject: {subject}\n"
            f"Form: {form or 'not identified'}\n"
            f"Chapter directory:\n{listing or 'Not available'}"
    }

# ---------------- KNOWLEDGE ----------------

INDEX = []
EXTENSIONS = {".txt",".md",".json"}

def clean(text):
    text = text.replace("\x00"," ")
    text = re.sub(r"[ \t]+"," ",text)
    text = re.sub(r"\n{3,}","\n\n",text)
    return text.strip()

def chunks(text, source, size=1600, overlap=250):
    text = clean(text)
    result = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))

        if end < len(text):
            boundary = text.rfind("\n", start + int(size*.6), end)
            if boundary > start:
                end = boundary

        part = text[start:end].strip()

        if part:
            result.append({
                "source": source,
                "text": part
            })

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return result

def rebuild_index():
    global INDEX
    INDEX = []
    KNOWLEDGE_DIR.mkdir(exist_ok=True)

    for path in KNOWLEDGE_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue

        try:
            text = path.read_text(encoding="utf-8")

            if path.suffix.lower() == ".json":
                try:
                    text = json.dumps(
                        json.loads(text),
                        ensure_ascii=False,
                        indent=2
                    )
                except Exception:
                    pass

            INDEX.extend(
                chunks(
                    text,
                    str(path.relative_to(KNOWLEDGE_DIR))
                )
            )

        except Exception as e:
            print("Knowledge file skipped:", path, e)

    print(f"Knowledge index ready: {len(INDEX)} chunks")

def words(text):
    return set(re.findall(r"[a-z0-9]+",(text or "").lower()))

def search_knowledge(prompt, subject, form="", limit=6):
    query = words(prompt)

    if not query:
        return []

    results = []

    for item in INDEX:
        body = words(item["text"])
        overlap = len(query & body)

        if overlap == 0:
            continue

        score = overlap / max(len(query),1)
        metadata = (
            item["source"] + " " +
            item["text"][:500]
        ).lower()

        if subject.lower() in metadata:
            score += .15

        if form and (
            f"form {form}" in metadata or
            f"tingkatan {form}" in metadata
        ):
            score += .10

        results.append((score,item))

    results.sort(reverse=True,key=lambda x:x[0])

    return [
        item for score,item in results[:limit]
        if score > 0
    ]

def source_text(results):
    if not results:
        return "No relevant local source was found."

    return "\n\n".join(
        f"[LOCAL SOURCE {i}: {x['source']}]\n{x['text']}"
        for i,x in enumerate(results,1)
    )

# ---------------- ROUTING ----------------

def detect_task(prompt, supplied=""):
    supplied = supplied.lower().strip()

    if supplied in TASKS:
        return supplied

    text = prompt.lower()

    for task, keywords in KEYWORDS.items():
        if any(x in text for x in keywords):
            return task

    return "explain"

def calculation(prompt):
    text = prompt.lower()

    patterns = [
        r"\bcalculate\b",
        r"\bwork out\b",
        r"\bhitungkan\b",
        r"\bfind the value\b",
        r"\bdetermine\b",
        r"\d+(?:\.\d+)?\s*(?:m/s|km/h|cm|mm|kg|mol|v|a|w|j|pa|hz|°c)\b"
    ]

    return any(re.search(x,text) for x in patterns)

def validate_image(value):
    if not isinstance(value,str):
        return None

    if not value.startswith("data:image/"):
        return None

    if ";base64," not in value:
        return None

    header, encoded = value.split(",",1)

    if not re.match(
        r"^data:image/(png|jpeg|jpg|webp|gif);base64$",
        header,
        re.I
    ):
        return None

    try:
        decoded = base64.b64decode(encoded,validate=True)
    except Exception:
        return None

    if len(decoded) > 8 * 1024 * 1024:
        return None

    return value

def history_clean(history):
    if not isinstance(history,list):
        return []

    result = []

    for item in history[-8:]:
        if not isinstance(item,dict):
            continue

        if item.get("role") not in ("user","assistant"):
            continue

        content = item.get("content")

        if isinstance(content,str) and content.strip():
            result.append({
                "role":item["role"],
                "content":content[:8000]
            })

    return result

def system_prompt(
    subject,
    level,
    language,
    task,
    prompt,
    curriculum,
    sources,
    image
):
    engine = SUBJECT_ENGINE.get(subject.lower(),"")
    chapter = curriculum.get("chapter")

    detected = (
        f"Chapter {chapter['chapter']}: {chapter['title']}"
        if chapter else
        "No exact chapter confidently detected."
    )

    image_instruction = ""

    if image:
        image_instruction = """
IMAGE MODE:
Carefully inspect the visible educational content.
Do not invent unreadable labels, values or text.
Separate visible information from scientific inference.
"""

    return f"""
{CORE}

{EXAM}

{TEACHING}

{engine}

TASK MODE:
{TASKS.get(task,TASKS["explain"])}

STUDENT CONTEXT:
Subject: {subject}
Level: {level}
Language: {language}
Question: {prompt}

CURRICULUM ROUTING:
{curriculum["context"]}
Detected location: {detected}

{image_instruction}

LOCAL KNOWLEDGE:
{source_text(sources)}

FINAL RULES:
- Answer the exact question.
- Explain WHY as well as WHAT when appropriate.
- Use correct scientific terminology.
- Use local source material for source-specific details.
- Never invent missing textbook information.
- Never fabricate official KPM/LPM material.
- Never fabricate page numbers.
- If evidence is insufficient, say so.
- Stay focused but teach properly.
- Answer in {language}.
"""

# ---------------- ROUTES ----------------

@app.route("/")
def home():
    index = BASE_DIR / "index.html"

    if index.exists():
        response = send_from_directory(str(BASE_DIR),"index.html")
        response.headers["Cache-Control"] = "no-cache"
        return response

    return jsonify({
        "success":True,
        "message":"Core Science AI backend is running."
    })

@app.route("/chat",methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}

        prompt = str(data.get("prompt","")).strip()

        if not prompt:
            return jsonify({
                "success":False,
                "error":"Please enter a question."
            }),400

        subject = normalise_subject(
            data.get("subject","Biology")
        )

        level = str(
            data.get("level","Form 4-5 / SPM")
        ).strip()

        language = str(
            data.get("language","English")
        ).strip() or "English"

        task = detect_task(
            prompt,
            data.get("task","")
        )

        if calculation(prompt):
            task = "solve"

        image = validate_image(
            data.get("image")
        )

        info = curriculum_info(
            subject,
            prompt,
            level
        )

        form = info["form"]

        sources = search_knowledge(
            prompt,
            subject,
            form
        )

        system = system_prompt(
            subject,
            level,
            language,
            task,
            prompt,
            info,
            sources,
            bool(image)
        )

        messages = [
            {
                "role":"system",
                "content":system
            }
        ]

        messages.extend(
            history_clean(
                data.get("history",[])
            )
        )

        if image:
            user_content = [
                {
                    "type":"text",
                    "text":prompt
                },
                {
                    "type":"image_url",
                    "image_url":{"url":image}
                }
            ]
        else:
            user_content = prompt

        messages.append({
            "role":"user",
            "content":user_content
        })

        model = VISION_MODEL if image else TEXT_MODEL

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=.2,
            max_tokens=4096
        )

        reply = (
            completion.choices[0]
            .message.content
            or ""
        )

        return jsonify({
            "success":True,
            "reply":reply,
            "task":task,
            "subject":subject,
            "level":level,
            "model":model,
            "routing":{
                "form":form or None,
                "chapter":info["chapter"],
                "retrieved_source_count":len(sources),
                "retrieved_sources":[
                    x["source"] for x in sources
                ]
            }
        })

    except Exception as e:
        print("CHAT ERROR:",repr(e))

        return jsonify({
            "success":False,
            "error":"The AI request failed. Check the Flask terminal."
        }),500

@app.route("/health")
def health():
    return jsonify({
        "status":"online",
        "service":"Core Science AI",
        "text_model":TEXT_MODEL,
        "vision_model":VISION_MODEL,
        "subjects":[
            "Biology",
            "Chemistry",
            "Physics"
        ],
        "knowledge_chunks":len(INDEX)
    })

@app.route("/curriculum")
def curriculum():
    return jsonify({
        "success":True,
        "curriculum":CURRICULUM
    })

@app.route("/knowledge/status")
def knowledge_status():
    files = []

    if KNOWLEDGE_DIR.exists():
        for path in KNOWLEDGE_DIR.rglob("*"):
            if (
                path.is_file() and
                path.suffix.lower() in EXTENSIONS
            ):
                files.append(
                    str(path.relative_to(KNOWLEDGE_DIR))
                )

    return jsonify({
        "success":True,
        "folder":"knowledge",
        "file_count":len(files),
        "chunk_count":len(INDEX),
        "files":files
    })

@app.route("/knowledge/reload",methods=["POST"])
def reload_knowledge():
    rebuild_index()

    return jsonify({
        "success":True,
        "message":"Knowledge index reloaded.",
        "chunk_count":len(INDEX)
    })

rebuild_index()

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
