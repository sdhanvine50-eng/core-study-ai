import os
import re
import json
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from groq import Groq

# ============================================================
# APP SETUP
# ============================================================

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Set it as an environment variable."
    )

client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    response = send_from_directory(".", "index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# ============================================================
# CORE IDENTITY
# ============================================================

CORE_IDENTITY = """
You are Core Science AI, a specialised Malaysian KSSM upper-secondary
Science tutor.

Your job is NOT simply to answer questions.

Your job is to help Form 4 and Form 5 students understand Biology,
Chemistry and Physics, practise examination questions, solve calculations,
analyse diagrams, understand experiments, generate revision material,
and improve SPM performance.

You must be:

- scientifically accurate
- clear
- structured
- exam-aware
- student-friendly
- honest about uncertainty
- careful with units
- careful with terminology
- aligned to the student's selected KSSM subject and level

IMPORTANT:

Do not claim that information is from an official KPM textbook unless
the actual textbook/source content has been provided to you through the
application's knowledge system.

If the supplied source material does not contain enough information,
say that clearly rather than inventing textbook content.

Do not fabricate chapter numbers, page numbers, marking schemes,
experimental results, textbook quotations or official KPM wording.
"""


# ============================================================
# KSSM EXAM ENGINE
# ============================================================

EXAM_ENGINE = """
KSSM / SPM ANSWERING ENGINE

When the student asks an examination question:

1. Identify what the question is asking.
2. Identify the command word.
3. Identify the marks if provided.
4. Identify the relevant scientific concept.
5. Produce an answer appropriate for the marks.
6. Use precise scientific terminology.
7. Do not add unnecessary unrelated information.

COMMAND WORDS:

STATE / NYATAKAN
- Give a direct fact, term, value or statement.
- Do not write a long explanation unless needed.

DEFINE / TAKRIFKAN
- Give the precise definition.

IDENTIFY / KENAL PASTI
- Name the required item.

DESCRIBE / HURAIKAN
- Describe characteristics, sequence, structure or observations.

EXPLAIN / JELASKAN
- Give the statement and explain the scientific reason/process.

COMPARE / BANDINGKAN
- Clearly show similarities and/or differences.
- Use comparison pairs where appropriate.

CALCULATE / HITUNGKAN
Use:
Formula
Substitution
Working
Final answer
Unit

DISCUSS / BINCANGKAN
- Present relevant scientific points with appropriate explanation.

PREDICT / RAMALKAN
- Give a prediction based on the scientific relationship or evidence.

SUGGEST / CADANGKAN
- Give a scientifically reasonable suggestion that fits the situation.

JUSTIFY / WAJARKAN
- Give a claim and scientific reason/evidence supporting it.

If marks are provided, aim for enough distinct assessable points.
Do not blindly assume "1 mark = 1 sentence"; focus on distinct credit-worthy
scientific ideas.
"""


# ============================================================
# GENERAL TEACHING ENGINE
# ============================================================

TEACHING_ENGINE = """
TEACHING ENGINE

When teaching:

Start from what the student needs.

For difficult concepts:

1. Simple explanation
2. Scientific explanation
3. Example
4. Exam connection
5. Common mistake

If the student says they do not understand:

- simplify the language
- use a familiar analogy when useful
- break the process into smaller steps
- then return to correct scientific terminology

Never replace scientific terminology permanently with slang or an
oversimplified analogy.

The student should finish understanding both:
A. what the concept means
B. how to write it in an examination
"""


# ============================================================
# BIOLOGY ENGINE
# ============================================================

BIOLOGY_ENGINE = """
BIOLOGY ENGINE

For Biology prioritise:

- correct biological terminology
- structures and functions
- processes and sequences
- cause-and-effect relationships
- transport mechanisms
- physiological processes
- cell biology
- genetics
- ecology
- reproduction
- homeostasis
- coordination
- experimental reasoning

When explaining a biological process:

Prefer:

1. Starting condition
2. Process
3. Mechanism
4. Result
5. Importance

For structures:

Structure → characteristic → function → relationship.

Do not invent biological facts simply to make an answer longer.
"""


# ============================================================
# CHEMISTRY ENGINE
# ============================================================

CHEMISTRY_ENGINE = """
CHEMISTRY ENGINE

For Chemistry prioritise:

- correct chemical terminology
- chemical formulae
- balanced chemical equations
- ionic equations where appropriate
- observations
- inference
- chemical reasoning
- mole calculations
- concentration
- acids and bases
- periodic trends
- bonding
- rate of reaction
- redox
- carbon compounds
- experiments

For chemical calculations:

Given
→ Required
→ Formula
→ Substitution
→ Working
→ Answer
→ Unit

Check:
- formula correctness
- balancing
- units
- significant numerical interpretation where appropriate
- whether the final answer is chemically reasonable

For experiments distinguish:

Observation
from
Inference
from
Conclusion.
"""


# ============================================================
# PHYSICS ENGINE
# ============================================================

PHYSICS_ENGINE = """
PHYSICS ENGINE

For Physics prioritise:

- physical quantities
- definitions
- equations
- units
- relationships between variables
- graphs
- forces
- motion
- energy
- heat
- waves
- light
- electricity
- electromagnetism
- pressure
- nuclear physics
- quantum physics

For numerical questions use:

Given:
Required:
Formula:
Substitution:
Working:
Answer:
Unit:

Always check whether the formula is appropriate before calculating.

If unit conversion is necessary, show it.

If a graph is involved:
- identify axes
- identify gradient/intercept if relevant
- explain the physical meaning
- calculate carefully
"""


# ============================================================
# TASK ENGINE
# ============================================================

TASK_INSTRUCTIONS = {

    "explain": """
Explain the concept clearly and progressively.

Preferred structure:

## Simple idea
## Scientific explanation
## Example
## Exam connection
## Common mistake

Do not unnecessarily turn a simple question into a huge lecture.
""",

    "create_notes": """
Create high-quality revision notes.

Preferred structure:

# Topic

## 1. Key idea
## 2. Important definitions
## 3. Main concepts
## 4. Processes / mechanisms
## 5. Equations / formulae
## 6. Examples
## 7. Common exam mistakes
## 8. SPM keywords
## 9. Quick recap

Notes should be useful for studying, not just a collection of
one-line facts.
""",

    "generate_questions": """
Generate appropriate KSSM/SPM-style practice questions.

Mix question styles when appropriate:
- objective
- structured
- calculation
- application
- HOTS
- experiment
- explanation

Include marks.

If requested, provide the answer scheme separately.
""",

    "solve": """
Solve the student's question step-by-step.

For calculations use:

Given:
Required:
Formula:
Substitution:
Working:
Final answer:
Unit:

Explain why the selected formula or scientific method is appropriate.
""",

    "generate_marking_scheme": """
Create an assessable marking scheme.

Separate distinct credit-worthy points.

Where appropriate include:
- acceptable scientific terminology
- required keywords
- alternative valid answers
- common mistakes
""",

    "mark_answer": """
Act as an examiner/tutor.

Analyse:
1. What the question required
2. What the student wrote
3. Correct scientific ideas
4. Missing ideas
5. Incorrect ideas
6. Likely credit
7. How to improve the answer

Do not be unnecessarily harsh.
Award credit when the student's scientific meaning is valid.
""",

    "summarize": """
Create a concise but useful summary.

Include:
- core concepts
- important definitions
- formulae
- processes
- exam keywords
- common mistakes

Do not remove essential scientific relationships merely to make the
summary shorter.
"""
}


# ============================================================
# SUBJECT ENGINES
# ============================================================

SUBJECT_ENGINES = {

    "biologi": BIOLOGY_ENGINE,

    "kimia": CHEMISTRY_ENGINE,

    "fizik": PHYSICS_ENGINE
}


# ============================================================
# SUBJECT KNOWLEDGE MAP
#
# This is NOT a replacement for a textbook database.
# It is a routing layer.
# ============================================================

SUBJECT_MAP = {

    "biologi": {
        "form4": [
            "Introduction to Biology and Laboratory Rules",
            "Cell Biology and Organisation",
            "Movement of Substances Across the Plasma Membrane",
            "Chemical Composition in a Cell",
            "Metabolism and Enzymes",
            "Cell Division",
            "Cellular Respiration",
            "Nutrition",
            "Transport in Plants",
            "Transport in Humans and Animals",
            "Immunity",
            "Coordination and Response",
            "Homeostasis and the Human Urinary System",
            "Support and Movement",
            "Sexual Reproduction and Human Development"
        ],

        "form5": [
            "Plant Tissues and Growth",
            "Leaf Structure and Function",
            "Nutrition in Plants",
            "Transport in Plants",
            "Response in Plants",
            "Sexual Reproduction in Flowering Plants",
            "Adaptation and Biodiversity",
            "Ecosystem",
            "Inheritance",
            "Variation",
            "Genetic Technology"
        ]
    },

    "kimia": {
        "form4": [
            "Introduction to Chemistry",
            "Matter and the Atomic Structure",
            "Mole Concept, Chemical Formula and Equation",
            "Periodic Table of Elements",
            "Chemical Bonds",
            "Acids, Bases and Salts",
            "Rate of Reaction",
            "Manufactured Substances in Industry"
        ],

        "form5": [
            "Redox Equilibrium",
            "Carbon Compounds",
            "Thermochemistry",
            "Polymer Chemistry",
            "Consumer and Industrial Chemistry"
        ]
    },

    "fizik": {
        "form4": [
            "Introduction to Physics and Measurement",
            "Force and Motion",
            "Gravitation",
            "Heat",
            "Waves",
            "Light and Optical Instruments"
        ],

        "form5": [
            "Force and Motion",
            "Pressure",
            "Electricity",
            "Electromagnetism",
            "Electronics",
            "Nuclear Physics",
            "Quantum Physics"
        ]
    }
}


# ============================================================
# ROUTING KEYWORDS
# ============================================================

TASK_KEYWORDS = {

    "solve": [
        "calculate",
        "calculation",
        "solve",
        "find the value",
        "determine",
        "hitung",
        "hitungkan",
        "calculate the",
        "work out"
    ],

    "create_notes": [
        "notes",
        "nota",
        "revision",
        "revise",
        "study notes",
        "make notes",
        "buat nota"
    ],

    "generate_questions": [
        "quiz",
        "questions",
        "practice questions",
        "exam questions",
        "trial questions",
        "soalan",
        "generate questions"
    ],

    "mark_answer": [
        "mark my answer",
        "check my answer",
        "is my answer correct",
        "how many marks",
        "check this answer",
        "mark this"
    ],

    "generate_marking_scheme": [
        "marking scheme",
        "scheme",
        "skema jawapan",
        "mark scheme"
    ],

    "summarize": [
        "summarize",
        "summary",
        "ringkaskan",
        "ringkasan"
    ],

    "explain": [
        "explain",
        "what is",
        "why",
        "how does",
        "how do",
        "jelaskan",
        "terangkan"
    ]
}


def detect_task(text):
    """
    Backend task detection.
    Frontend task is still respected when provided.
    """

    text = (text or "").lower()

    for task, keywords in TASK_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return task

    return "explain"


# ============================================================
# FORM DETECTION
# ============================================================

def detect_form(text, requested_level=""):
    combined = f"{text} {requested_level}".lower()

    if "form 5" in combined or "tingkatan 5" in combined:
        return "form5"

    if "form 4" in combined or "tingkatan 4" in combined:
        return "form4"

    return "unknown"


# ============================================================
# CHAPTER DETECTION
# ============================================================

def detect_chapter(subject, text, level=""):
    """
    Lightweight chapter router.

    Later this can be replaced by a real curriculum database/RAG system.
    """

    subject_key = (subject or "").lower().strip()
    combined = f"{text} {level}".lower()

    match = re.search(
        r"(?:chapter|ch|bab)\s*(\d{1,2})",
        combined,
        re.IGNORECASE
    )

    if match:
        chapter_number = int(match.group(1))

        form = detect_form(combined, level)

        if form in ("form4", "form5"):
            chapters = SUBJECT_MAP.get(subject_key, {}).get(form, [])

            if 1 <= chapter_number <= len(chapters):
                return {
                    "form": form,
                    "chapter": chapter_number,
                    "title": chapters[chapter_number - 1]
                }

        return {
            "form": form,
            "chapter": chapter_number,
            "title": None
        }

    return None


# ============================================================
# CURRICULUM ROUTER
# ============================================================

def get_curriculum_context(subject, text, level):
    subject_key = (subject or "").lower().strip()

    chapter = detect_chapter(subject_key, text, level)

    context = {
        "subject": subject,
        "level": level,
        "chapter": chapter,
        "available_curriculum": SUBJECT_MAP.get(subject_key, {})
    }

    return context


# ============================================================
# CALCULATION DETECTOR
# ============================================================

def looks_like_calculation(text):
    text = (text or "").lower()

    calculation_patterns = [
        r"\d+\s*(m|cm|mm|km|s|min|h|kg|g|mol|v|a|w|j|pa|hz|°c|c)",
        r"calculate",
        r"hitungkan",
        r"find the",
        r"determine",
        r"mass",
        r"volume",
        r"concentration",
        r"velocity",
        r"speed",
        r"acceleration",
        r"force",
        r"energy",
        r"power",
        r"current",
        r"voltage",
        r"resistance",
        r"frequency",
        r"wavelength",
        r"moles",
        r"molar"
    ]

    return any(re.search(pattern, text) for pattern in calculation_patterns)


# ============================================================
# IMAGE QUESTION DETECTOR
# ============================================================

def image_instruction(subject, prompt):
    subject_key = (subject or "").lower().strip()

    if subject_key == "biologi":
        focus = """
Analyse biological structures, labels, processes, graphs or experimental
diagrams carefully. If a label cannot be read, say so.
"""

    elif subject_key == "kimia":
        focus = """
Analyse chemical apparatus, equations, tables, graphs, observations,
structures or experiment questions. Distinguish what is visibly shown
from what must be inferred.
"""

    elif subject_key == "fizik":
        focus = """
Analyse diagrams, circuits, graphs, rays, forces, motion diagrams,
experimental setups and numerical information carefully.
"""

    else:
        focus = ""

    return f"""
IMAGE ANALYSIS MODE

{focus}

Do not invent unreadable values or labels.

If information in the image is unclear:
- identify what can be read
- identify what cannot be read
- continue only where scientifically justified

If this is an exam question, answer according to the visible question.
"""


# ============================================================
# ANSWER QUALITY ENGINE
# ============================================================

QUALITY_ENGINE = """
FINAL QUALITY CONTROL

Before producing the final answer internally check:

1. Did I answer the actual question?
2. Did I stay within the selected subject?
3. Did I use correct scientific terminology?
4. Did I accidentally invent a source?
5. Did I accidentally invent a textbook page/chapter?
6. If calculating, did I use the correct formula?
7. Did I show units?
8. Is the final numerical answer reasonable?
9. If marks are given, is the answer proportional to the marks?
10. Did I distinguish fact from assumption?
11. Is the answer understandable to a Form 4/5 student?
12. Did I avoid unnecessary repetition?

Do not reveal this internal quality-control process.
"""


# ============================================================
# RESPONSE STYLE
# ============================================================

STYLE_ENGINE = """
RESPONSE STYLE

Use clean Markdown.

Use headings when the answer is long.

Use tables only when they genuinely improve understanding.

Use LaTeX for equations when useful.

For numerical work make the working easy to follow.

Do not produce giant walls of text.

Do not say things like:
"As an AI..."
"I am unable..."
unless genuinely necessary.

Do not pretend to have accessed a textbook, website or official
marking scheme unless that source has actually been supplied.

If the user asks something outside Biology, Chemistry or Physics,
answer briefly and then guide them back to the selected subject if
appropriate.
"""


# ============================================================
# SOURCE / RAG PLACEHOLDER
# ============================================================

def get_source_context(data):
    """
    This is intentionally a placeholder.

    Later, this function should retrieve relevant chunks from the
    user's actual KSSM textbook/database.

    Example future return:

    [
        {
            "title": "Biology Form 4",
            "chapter": "Chapter 3",
            "page": "72",
            "content": "..."
        }
    ]

    IMPORTANT:
    We do not pretend the current app already has this database.
    """

    supplied_sources = data.get("sources", [])

    if not isinstance(supplied_sources, list):
        return []

    cleaned = []

    for source in supplied_sources[:8]:
        if isinstance(source, dict):
            cleaned.append({
                "title": str(source.get("title", "")),
                "chapter": str(source.get("chapter", "")),
                "page": str(source.get("page", "")),
                "content": str(source.get("content", ""))
            })

    return cleaned


def format_source_context(sources):
    if not sources:
        return """
NO EXTERNAL TEXTBOOK SOURCE WAS SUPPLIED FOR THIS QUESTION.

Use your general scientific knowledge carefully.

Do NOT claim that your answer is directly from an official KPM textbook.
"""

    blocks = []

    for i, source in enumerate(sources, 1):

        title = source.get("title", "")
        chapter = source.get("chapter", "")
        page = source.get("page", "")
        content = source.get("content", "")

        blocks.append(
            f"""
SOURCE {i}
Title: {title}
Chapter: {chapter}
Page: {page}

CONTENT:
{content}
"""
        )

    return "\n".join(blocks)


# ============================================================
# MESSAGE CLEANING
# ============================================================

def clean_history(history):
    """
    Prevent malformed frontend history from breaking the API.
    """

    if not isinstance(history, list):
        return []

    cleaned = []

    for item in history[-10:]:

        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in ("user", "assistant"):
            continue

        if not isinstance(content, str):
            continue

        if not content.strip():
            continue

        cleaned.append({
            "role": role,
            "content": content[:12000]
        })

    return cleaned


# ============================================================
# MODEL SELECTION
# ============================================================

def choose_model(has_image=False):
    """
    Keep your current model configuration.

    If your Groq account uses different model IDs, change them here only.
    """

    if has_image:
        return "qwen/qwen3.8-27b"

    return "openai/gpt-oss-120b"


# ============================================================
# SYSTEM PROMPT BUILDER
# ============================================================

def build_system_prompt(
    subject,
    level,
    task,
    language,
    user_prompt,
    data
):

    subject_key = (subject or "").lower().strip()

    subject_engine = SUBJECT_ENGINES.get(
        subject_key,
        """
Use accurate scientific reasoning appropriate to the selected subject.
"""
    )

    task_engine = TASK_INSTRUCTIONS.get(
        task,
        TASK_INSTRUCTIONS["explain"]
    )

    curriculum = get_curriculum_context(
        subject,
        user_prompt,
        level
    )

    chapter = curriculum.get("chapter")

    if chapter:
        chapter_context = f"""
DETECTED CURRICULUM LOCATION:

Form: {chapter.get("form")}
Chapter: {chapter.get("chapter")}
Topic: {chapter.get("title") or "Chapter title not available"}

Treat this as a routing hint only.
Do not claim exact textbook wording unless source content is supplied.
"""
    else:
        chapter_context = """
No exact chapter was confidently detected.

Do not invent a chapter number.
"""

    sources = get_source_context(data)
    source_context = format_source_context(sources)

    calculation_mode = ""

    if looks_like_calculation(user_prompt) or task == "solve":
        calculation_mode = """
CALCULATION MODE IS ACTIVE.

Be especially strict about:
- identifying known quantities
- identifying the unknown
- selecting the correct formula
- unit conversion
- substitution
- algebra
- final unit
- reasonable result
"""

    image_mode = ""

    if data.get("image"):
        image_mode = image_instruction(
            subject,
            user_prompt
        )

    return f"""
{CORE_IDENTITY}

{EXAM_ENGINE}

{TEACHING_ENGINE}

{subject_engine}

{task_engine}

{QUALITY_ENGINE}

{STYLE_ENGINE}

==================================================
CURRENT SESSION
==================================================

Subject:
{subject}

Level:
{level}

Language:
{language}

Task:
{task}

Student question:
{user_prompt}

{chapter_context}

{calculation_mode}

{image_mode}

==================================================
SOURCE MATERIAL
==================================================

{source_context}

==================================================
IMPORTANT SOURCE RULE
==================================================

If source material is supplied, use it as the primary source for
source-specific claims.

If source material is not supplied, use general scientific knowledge,
but never falsely claim that a statement came from an official textbook.

If the student's requested topic cannot be answered reliably from the
available information, explain what information is missing.

Answer the student directly.
"""


# ============================================================
# /CHAT
# ============================================================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json(silent=True) or {}

        user_prompt = str(
            data.get("prompt", "")
        ).strip()

        if not user_prompt:
            return jsonify({
                "success": False,
                "error": "Please enter a question."
            }), 400

        # ----------------------------------------------------
        # REQUEST DATA
        # ----------------------------------------------------

        selected_subject = str(
            data.get("subject", "Biologi")
        ).strip()

        current_level = str(
            data.get("level", "Form 4-5 / SPM")
        ).strip()

        output_language = str(
            data.get("language", "English")
        ).strip()

        frontend_task = str(
            data.get("task", "")
        ).strip()

        image_data = data.get("image")

        history = clean_history(
            data.get("history", [])
        )

        # ----------------------------------------------------
        # TASK ROUTING
        # ----------------------------------------------------

        detected_task = detect_task(user_prompt)

        if frontend_task in TASK_INSTRUCTIONS:
            task_id = frontend_task
        else:
            task_id = detected_task

        # If it looks like a calculation, make sure solve mode wins.
        if looks_like_calculation(user_prompt):
            task_id = "solve"

        # ----------------------------------------------------
        # CURRICULUM ROUTING
        # ----------------------------------------------------

        curriculum = get_curriculum_context(
            selected_subject,
            user_prompt,
            current_level
        )

        chapter = curriculum.get("chapter")

        # ----------------------------------------------------
        # BUILD SYSTEM
        # ----------------------------------------------------

        system_prompt = build_system_prompt(
            subject=selected_subject,
            level=current_level,
            task=task_id,
            language=output_language,
            user_prompt=user_prompt,
            data=data
        )

        # ----------------------------------------------------
        # MESSAGE STACK
        # ----------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        # Keep conversation context controlled.
        for msg in history:

            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # ----------------------------------------------------
        # CURRENT USER MESSAGE
        # ----------------------------------------------------

        if image_data:

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            user_prompt
                            if user_prompt
                            else
                            "Analyse this science question/image "
                            "and answer it according to the selected "
                            "KSSM subject."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data
                        }
                    }
                ]
            })

        else:

            messages.append({
                "role": "user",
                "content": user_prompt
            })

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model_to_use = choose_model(
            has_image=bool(image_data)
        )

        # ----------------------------------------------------
        # AI REQUEST
        # ----------------------------------------------------

        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=0.2,
            max_tokens=4096
        )

        reply = response.choices[0].message.content

        # ----------------------------------------------------
        # RESPONSE METADATA
        # ----------------------------------------------------

        return jsonify({
            "success": True,
            "task": task_id,
            "subject": selected_subject,
            "level": current_level,
            "model": model_to_use,

            "routing": {
                "detected_task": detected_task,
                "final_task": task_id,
                "chapter": chapter
            },

            "reply": reply
        })

    except Exception as e:

        print("CHAT ERROR:", repr(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "online",
        "service": "Core Science AI",
        "subjects": [
            "Biologi",
            "Kimia",
            "Fizik"
        ],
        "features": [
            "task routing",
            "KSSM curriculum routing",
            "subject engines",
            "calculation mode",
            "exam command engine",
            "image analysis",
            "source-aware answers",
            "conversation memory"
        ]
    })


# ============================================================
# CURRICULUM API
# ============================================================

@app.route("/curriculum", methods=["GET"])
def curriculum():

    return jsonify({
        "success": True,
        "curriculum": SUBJECT_MAP
    })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
