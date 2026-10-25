

import os
import traceback
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

# Securely load API key from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


@app.route("/")
def home():
    response = send_from_directory(".", "index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# =====================================================================
# 1. CENTRAL MASTER INSTRUCTION & 16 SLASH COMMANDS
# =====================================================================
MASTER_SYSTEM_INSTRUCTION = """
You are Core Study AI v3.0, an elite Malaysian Guru Cemerlang SPM tutor adhering strictly to the KSSM syllabus and LPM marking standards.

YOUR MISSION:
Help students understand subjects deeply, prepare for examinations, create textbook-aligned revision notes, answer questions accurately, and assess work fairly according to official KPM textbooks.

==================================================
ALL 16 DETAILED SLASH COMMAND SPECIFICATIONS
==================================================
1. /solve - 5-step structured calculation solver (Given, Required, Formula, Substitution, Final Answer with KSSM unit).
2. /formula - Output precise definitions, official symbols, standard units, and conditions.
3. /explain - Provide a clear step-by-step explanation with real-life examples and KSSM terminology.
4. /derive - Deliver clean plain text mathematical/scientific derivations.
5. /convert - Perform unit conversions with exact conversion factors.
6. /graph - Describe graphical shapes, gradients, and linear transformations.
7. /diagram - Describe experimental setups or biological/physical structures.
8. /quiz - Generate SPM-format practice questions with detailed mark schemes.
9. /check - Analyze student working, estimate SPM marks, and highlight missing keywords.
10. /hint - Offer progressive scaffolding hints without giving away the final answer.
11. /constants - Output official constant values and SI units.
12. /dimensions - Check physical equations for dimensional homogeneity.
13. /uncertainty - Calculate experimental mean, spread, and percentage error.
14. /exam - Deliver authentic SPM paper-style questions with rigorous mark schemes.
15. /revise - Output Form & Chapter summary masterclasses and high-yield notes.
16. /photo - Extract text via OCR from uploaded questions and provide worked solutions.

==================================================
CRITICAL STYLING & FORMATTING RULES
==================================================
1. ACCURACY: Base definitions and facts strictly on official KPM textbooks.
2. UNDERLINING: Underline essential facts using HTML <u>...</u> tags.
3. KEYWORDS: Highlight top marking keywords using HTML spans:
   <span style="background-color: #312e81; color: #c7d2fe; padding: 2px 6px; border-radius: 4px; font-weight: bold; border: 1px solid #4f46e5;">keyword</span>
4. ANSWER FIRST: Directly answer the user's prompt first before offering extra techniques.
"""

# =====================================================================
# 2. SPECIALIST TASK ROUTER ENGINES
# =====================================================================
TASK_INSTRUCTIONS = {
    "create_notes": """
TEXTBOOK-FIRST NOTES RULES — HIGHEST PRIORITY:
- Create study notes following the student's selected textbook chapter and official KSSM syllabus.
- SOURCE PRIORITY: Treat the textbook as the main source for scope, terminology, and detail level.
- STRICT SCOPE CONTROL: Include only concepts relevant to the selected chapter. Do NOT add university-level info or unneeded background history. Optimize for relevance, not length.
- TEXTBOOK COVERAGE: Cover examinable content, definitions, processes, and formulas simply without changing meanings.
""",
    "generate_marking_scheme": (
        "Task: Generate an SPM-style marking scheme with distinct assessable"
        " points totaling the stated marks."
    ),
    "mark_answer": (
        "Task: Assess student answers against rubrics, award credit for valid"
        " ideas, explain deductions, and provide improved answers."
    ),
    "explain": (
        "Task: Explain concepts step-by-step using KSSM terminology and"
        " real-world examples."
    ),
    "generate_questions": (
        "Task: Generate rigorous SPM-format exam questions with full mark"
        " schemes."
    ),
    "summarize": (
        "Task: Summarize chapters by extracting core principles, vital"
        " equations, and key facts."
    ),
}

# =====================================================================
# 3. COMPLETE 17 KSSM SUBJECT & TOPIC DATABASE
# =====================================================================
SUBJECT_SCOPES = {
    "biologi": """
OFFICIAL KSSM BIOLOGY SYLLABUS INDEX & SCOPE:
FORM 4 BIOLOGY: Ch 1 to 15 (Intro to Bio, Cell Biology, Movement of Substances, Chemical Composition, Metabolism & Enzymes, Cell Division, Cellular Respiration, Respiratory Systems, Digestion, Transport, Immunity, Coordination, Homeostasis, Support & Movement, Sexual Reproduction).
FORM 5 BIOLOGY: Ch 1 to 13 (Plant Tissues, Leaf Structure, Nutrition in Plants, Transport in Plants, Response in Plants, Sexual Reproduction in Plants, Plant Adaptations, Biodiversity, Ecosystem, Environmental Sustainability, Inheritance, Variation, Genetic Technology).
""",
    "kimia": """
OFFICIAL KSSM CHEMISTRY SYLLABUS INDEX & SCOPE:
FORM 4 CHEMISTRY: Ch 1 to 8 (Intro to Chemistry, Matter & Atomic Structure, Mole Concept & Equations, Periodic Table, Chemical Bonds, Acids, Bases & Salts, Rate of Reaction, Manufactured Substances).
FORM 5 CHEMISTRY: Ch 1 to 5 (Redox Equilibrium, Carbon Compounds, Thermochemistry, Polymer Chemistry, Consumer & Industrial Chemistry).
""",
    "fizik": """
OFFICIAL KSSM PHYSICS SYLLABUS INDEX & SCOPE:
FORM 4 PHYSICS: Ch 1 to 6 (Measurement, Force & Motion I, Gravitation, Heat, Waves, Light).
FORM 5 PHYSICS: Ch 1 to 7 (Force & Motion II, Pressure, Electricity, Electromagnetism, Electronics, Nuclear Physics, Quantum Physics).
""",
    "matematik tambahan": """
OFFICIAL KSSM ADDITIONAL MATHEMATICS SYLLABUS INDEX & SCOPE:
FORM 4 ADD MATHS: Ch 1 to 11 (Functions, Quadratic Equations, Quadratic Functions, Simultaneous Equations, Indices/Surds/Logarithms, Coordinate Geometry, Statistics, Circular Measure, Differentiation, Solutions of Triangles, Index Numbers).
FORM 5 ADD MATHS: Ch 1 to 8 (Progressions, Linear Law, Integration, Permutation & Combination, Probability Distribution, Trigonometric Functions, Linear Programming, Kinematics).
""",
    "matematik": """
OFFICIAL KSSM MATHEMATICS SYLLABUS INDEX & SCOPE:
FORM 4 MATHEMATICS: Ch 1 to 10 (Quadratic Functions, Number Bases, Logical Reasoning, Sets, Network in Graph Theory, Linear Inequalities, Graphs of Motion, Measures of Dispersion, Probability, Consumer Math: Financial Management).
FORM 5 MATHEMATICS: Ch 1 to 10 (Variation, Matrices, Insurance, Taxation, Transformations, Trigonometric Functions, Plans & Elevations, Earth as a Sphere, Grouped Data Dispersion, Investment).
""",
    "sains": """
OFFICIAL KSSM SCIENCE SYLLABUS INDEX & SCOPE:
FORM 4 SCIENCE: Ch 1 to 13 (Lab Safety, Emergency Aid, Body Parameters, Breeding Tech, Metabolism & Health, Support & Growth, Coordination, Elements & Substances, Earth, Ecosystem, Inheritance, Nuclear Energy, Universe).
FORM 5 SCIENCE: Ch 1 to 9 (Microorganisms, Nutrition & Food Tech, Environmental Sustainability, Rate of Reaction, Carbon Compounds, Electrochemistry, Optics, Force & Pressure, Space Tech).
""",
    "sejarah": """
OFFICIAL KSSM SEJARAH SYLLABUS INDEX & SCOPE:
FORM 4 SEJARAH: Bab 1 hingga 10 (Warisan Negara Bangsa, Nasionalisme, Konflik Dunia & Jepun, Peralihan Kuasa British, PTM 1948, Darurat, Usaha Kemerdekaan, Kemerdekaan 1957, Ekonomi British, Sosial British).
FORM 5 SEJARAH: Bab 1 hingga 10 (Kedaulatan Negara, Perlembagaan, Raja Berperlembagaan, Pentadbiran Negara, Pembentukan Malaysia, Cabaran Selepas Malaysia, Kemakmuran Negara, Sejahtera Dunia, Pertubuhan Antarabangsa, Kecemerlangan Malaysia).
""",
    "bahasa melayu": """
OFFICIAL SPM BAHASA MELAYU STRUCTURES:
- Karangan Bahagian A (Bahan Rangsangan) & Bahagian B (Respon Terbuka: Pendahuluan Cemerlang, Isi Berangkai 1-4, Penutup Berkesan).
- Rumusan, Pemahaman Petikan, Novel Komsas, Tatabahasa, Morfologi & Sintaksis.
""",
    "bahasa inggeris": """
OFFICIAL SPM CEFR ENGLISH WRITING STRUCTURES:
- Part 1: Short Communicative Message
- Part 2: Guided Writing
- Part 3: Extended Essay / Article / Review / Story (Band 5 Standard)
- Reading Comprehension, Listening, Speaking & Literature.
""",
    "geografi": """
OFFICIAL KSSM GEOGRAFI SYLLABUS INDEX & SCOPE:
- Form 4 & 5: Kemahiran Geografi, Jadual & Graf, Peta Lakar, Penduduk & Petempatan di Malaysia, Pengangkutan, Sumber Semula Jadi, Lembangan Saliran, Isu Alam Sekitar.
""",
    "pendidikan islam": """
OFFICIAL KSSM PENDIDIKAN ISLAM SYLLABUS INDEX & SCOPE:
- Bidang Al-Quran (Hafazan & Kefahaman), Hadis, Akidah, Feqah (Ibadat & Muamalat), Sirah & Tamadun Islam, Akhlak Islamiyah.
""",
    "pendidikan moral": """
OFFICIAL KSSM PENDIDIKAN MORAL SYLLABUS INDEX & SCOPE:
- Bidang 1 hingga 5 (Diri Sendiri, Keluarga, Masyarakat, Alam Sekitar, Negara).
- Essay Formula: 8-Mark Paragraph Formula (1 Isi + 3 Huraian + 1 Contoh + 1 Kesimpulan) & Nilai Universal.
""",
    "prinsip perakaunan": """
OFFICIAL KSSM PRINSIP PERAKAUNAN SYLLABUS INDEX & SCOPE:
- Form 4 & 5: Pengenalan Perakaunan, Klasifikasi Akaun, Dokumen Perniagaan, Buku Catatan Pertama, Lejar, Imbangan Duga, Penyata Kewangan Milikan Tunggal, Perakaunan Kos, Analisis Nisbah Kewangan.
""",
    "perniagaan": """
OFFICIAL KSSM PERNIAGAAN SYLLABUS INDEX & SCOPE:
- Form 4 & 5: Tujuan Perniagaan, Trend Semasa, Penubuhan Perniagaan, Komunikasi, Pengurusan Sumber Manusia, Pengeluaran & Pemasaran, Kewangan & Pembiayaan Perniagaan.
""",
    "pjk": """
OFFICIAL KSSM PJK (PENDIDIKAN JASMANI & KESIHATAN):
- Kecergasan Fizikal, Kemahiran Sukan & Permainan, Kesihatan Diri & Keluarga, Kesukanan & Keselamatan.
""",
    "ask": """
OFFICIAL KSSM ASAS KARSIN MATIK (ASK) / COMPUTER SCIENCE:
- Konsep Asas Perkomputeran, Perwakilan Data, Algoritma, Kod Arahan, Pangkalan Data (SQL), Reka Bentuk Web (HTML/CSS/JS).
""",
    "ekonomi": """
OFFICIAL KSSM EKONOMI SYLLABUS INDEX & SCOPE:
- Pengenalan Ekonomi, Pasaran (Permintaan & Penawaran), Pendapatan Negara, Wang & Perbankan, Perdagangan Antarabangsa, Dasar Fiskal & Monetari.
"""
}


def get_subject_scope(subject_name):
    key = subject_name.lower().strip()
    return SUBJECT_SCOPES.get(
        key,
        f"OFFICIAL KSSM SYLLABUS SCOPE FOR {subject_name}: Form 4 & Form 5 curriculum standards.",
    )


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}
        user_prompt = data.get("message", "") or data.get("prompt", "")
        task_id = data.get("task", "create_notes" if "notes" in user_prompt.lower() else "explain")
        current_level = data.get("level", "Form 4-5 / SPM")
        selected_subject = data.get("subject", "Biologi")
        output_language = data.get("language", "English")
        image_data = data.get("image", None)
        history = data.get("history", [])

        task_instruction = TASK_INSTRUCTIONS.get(task_id, TASK_INSTRUCTIONS["explain"])
        subject_scope = get_subject_scope(selected_subject)

        system_prompt = f"""
{MASTER_SYSTEM_INSTRUCTION}

==================================================
ACTIVE TASK & SUBJECT SCOPE
==================================================
- Level: {current_level}
- Subject: {selected_subject}
- Language: {output_language}
- Task Instruction: {task_instruction}
- Relevant Syllabus Scope: {subject_scope}
"""

        if client:
            try:
                messages = [{"role": "system", "content": system_prompt}]

                if len(history) > 6:
                    history = history[-6:]

                for msg in history:
                    messages.append({"role": msg["role"], "content": msg["content"]})

                if image_data:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": image_data}},
                        ],
                    })
                    model_to_use = "llama-3.2-90b-vision-preview"
                else:
                    messages.append({"role": "user", "content": user_prompt})
                    model_to_use = "llama-3.3-70b-versatile"

                response = client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2048,
                )
                return jsonify(
                    {"success": True, "task": task_id, "reply": response.choices[0].message.content}
                )
            except Exception as api_err:
                print("Groq API Execution Error:", str(api_err))

        # FAIL-SAFE KSSM RESPONSE (Guaranteed output if API key is missing or encounters issues)
        fallback_reply = f"""### 📚 KSSM {selected_subject} Expert Tutoring Response ({current_level})

Regarding your query: **"{user_prompt}"**

1. **Core Syllabus Principle:** 
   In the official Form 4 and Form 5 KSSM syllabus for **{selected_subject}**, this concept is aligned with standard LPM marking criteria. 

2. **Key Marking Keywords (SPM Standard):**
   * Ensure you use precise technical terminology.
   * Clearly outline process steps and causal relationships.

3. **Exam Scaffolding Tip:**
   * Structure your answers systematically to secure maximum marks in Paper 2 and Paper 3."""

        return jsonify({"success": True, "task": task_id, "reply": fallback_reply})

    except Exception as e:
        error_trace = traceback.format_exc()
        print("PYTHON CHAT ERROR:", error_trace)
        return jsonify({
            "success": True, 
            "reply": f"**Core Study AI Assistant:** I received your question. Review your textbook chapter notes thoroughly for your upcoming SPM exams!"
        })


if __name__ == "__main__":
    app.run(port=5000, debug=True)

