
import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

# Securely load API key from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


@app.route("/")
def home():
    response = send_from_directory(".", "index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# =====================================================================
# 1. CENTRAL MASTER INSTRUCTION & 8-LAYER KSSM ARCHITECTURE (FORM 4-5)
# =====================================================================
MASTER_SYSTEM_INSTRUCTION = """
You are Core Study AI v3.0, an elite Malaysian Guru Cemerlang SPM tutor powered by an 8-Layer KSSM Neural Architecture.

YOUR MISSION:
Help Upper Secondary students (Form 4 and Form 5) understand subjects deeply, prepare for SPM examinations, create textbook-aligned revision notes, answer questions accurately, and assess work strictly according to official KPM textbooks and LPM standards.

==================================================
THE 8-LAYER KSSM COGNITIVE ARCHITECTURE YOU MUST FOLLOW:
==================================================
1. CURRICULUM DATABASE: Strictly align all responses with KSSM Form 4–5 standards, DSKP learning objectives, and official chapter scopes.
2. VERIFIED KNOWLEDGE: Base all definitions, concepts, and formulas strictly on official KPM textbooks—avoid university-level fluff.
3. QUESTION BANK ALIGNMENT: Categorize and structure content matching Objective, Structured, Essay, Calculation, or HOTS formats.
4. MARKING-SCHEME ENGINE: Map answers directly to mark allocations. (e.g., 4 marks = 4 explicit credit points using exact KSSM terminology).
5. COMMAND-WORD ENGINE: Obey Malaysian exam command words strictly:
   - Nyatakan / State -> Direct, precise answer.
   - Jelaskan / Explain -> Point + logical scientific/mathematical explanation.
   - Huraikan / Describe -> Detailed chronological or structural description.
   - Bandingkan / Compare -> Explicit similarities and differences.
   - Hitungkan / Calculate -> Formula -> Substitution -> Working -> Final Answer with correct SI Unit.
6. CALCULATION ENGINE: For Maths, Add Maths, Physics, and Chemistry, always show full working steps clearly.
7. TUTOR ENGINE: Adapt dynamically. If a student says "I don't understand", simplify or provide a real-world KSSM-friendly analogy.
8. STUDENT SYSTEM MINDSET: Focus heavily on high-yield exam scoring, spotting weak links, and teaching marking keywords.

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
# 3. COMPLETE 17 KSSM SUBJECT & ACCURATE CHAPTER DATABASE
# =====================================================================
SUBJECT_SCOPES = {
    "biologi": """
OFFICIAL KSSM BIOLOGY SYLLABUS INDEX & SCOPE:
FORM 4 BIOLOGY:
- Chapter 1: Introduction to Biology and Laboratory Rules
- Chapter 2: Cell Biology and Organisation
- Chapter 3: Movement of Substances Across a Plasma Membrane
- Chapter 4: Chemical Composition of a Cell
- Chapter 5: Metabolism and Enzymes
- Chapter 6: Cell Division
- Chapter 7: Cellular Respiration
- Chapter 8: Respiratory Systems in Humans and Animals
- Chapter 9: Nutrition and the Human Digestive System
- Chapter 10: Transport in Humans and Animals
- Chapter 11: Immunity in Humans
- Chapter 12: Coordination and Response in Humans
- Chapter 13: Homeostasis and the Human Urinary System
- Chapter 14: Support and Movement in Humans and Animals
- Chapter 15: Sexual Reproduction, Development and Growth
FORM 5 BIOLOGY:
- Chapter 1: Organisation of Plant Tissues and Growth
- Chapter 2: Leaf Structure and Function
- Chapter 3: Nutrition in Plants
- Chapter 4: Transport in Plants
- Chapter 5: Response in Plants
- Chapter 6: Sexual Reproduction in Flowering Plants
- Chapter 7: Adaptations of Plants in Different Habitats
- Chapter 8: Biodiversity
- Chapter 9: Ecosystem
- Chapter 10: Environmental Sustainability
- Chapter 11: Inheritance
- Chapter 12: Variation
- Chapter 13: Genetic Technology
""",
    "kimia": """
OFFICIAL KSSM CHEMISTRY SYLLABUS INDEX & SCOPE:
FORM 4 CHEMISTRY:
- Chapter 1: Introduction to Chemistry
- Chapter 2: Matter and Atomic Structure
- Chapter 3: The Mole Concept, Chemical Formulae and Equations
- Chapter 4: The Periodic Table of Elements
- Chapter 5: Chemical Bond
- Chapter 6: Acid, Base and Salt
- Chapter 7: Rate of Reaction
- Chapter 8: Manufactured Substances in Industry
FORM 5 CHEMISTRY:
- Chapter 1: Redox Equilibrium
- Chapter 2: Carbon Compound
- Chapter 3: Thermochemistry
- Chapter 4: Polymer Chemistry
- Chapter 5: Consumer and Industrial Chemistry
""",
    "fizik": """
OFFICIAL KSSM PHYSICS SYLLABUS INDEX & SCOPE:
FORM 4 PHYSICS:
- Chapter 1: Measurement
- Chapter 2: Force and Motion I
- Chapter 3: Gravitation
- Chapter 4: Heat
- Chapter 5: Waves
- Chapter 6: Light
FORM 5 PHYSICS:
- Chapter 1: Force and Motion II
- Chapter 2: Pressure
- Chapter 3: Electricity
- Chapter 4: Electromagnetism
- Chapter 5: Electronics
- Chapter 6: Nuclear Physics
- Chapter 7: Quantum Physics
""",
    "matematik tambahan": """
OFFICIAL KSSM ADDITIONAL MATHEMATICS SYLLABUS INDEX & SCOPE:
FORM 4 ADD MATHS:
- Chapter 1: Functions
- Chapter 2: Quadratic Equations
- Chapter 3: Quadratic Functions
- Chapter 4: Simultaneous Equations
- Chapter 5: Indices, Surds and Logarithms
- Chapter 6: Coordinate Geometry
- Chapter 7: Statistics
- Chapter 8: Circular Measure
- Chapter 9: Differentiation
- Chapter 10: Solutions of Triangles
- Chapter 11: Index Numbers
FORM 5 ADD MATHS:
- Chapter 1: Progressions
- Chapter 2: Linear Law
- Chapter 3: Integration
- Chapter 4: Permutation and Combination
- Chapter 5: Probability Distribution
- Chapter 6: Trigonometric Functions
- Chapter 7: Linear Programming
- Chapter 8: Kinematics of Linear Motion
""",
    "matematik": """
OFFICIAL KSSM MATHEMATICS SYLLABUS INDEX & SCOPE:
FORM 4 MATHEMATICS:
- Chapter 1: Quadratic Functions and Equations in One Variable
- Chapter 2: Number Bases
- Chapter 3: Logical Reasoning
- Chapter 4: Operations on Sets
- Chapter 5: Network in Graph Theory
- Chapter 6: Linear Inequalities in Two Variables
- Chapter 7: Graphs of Motion
- Chapter 8: Measures of Dispersion for Ungrouped Data
- Chapter 9: Probability of Combined Events
- Chapter 10: Consumer Mathematics: Financial Management
FORM 5 MATHEMATICS:
- Chapter 1: Variation
- Chapter 2: Matrices
- Chapter 3: Consumer Mathematics: Insurance
- Chapter 4: Consumer Mathematics: Taxation
- Chapter 5: Congruency, Enlargement and Combination of Transformations
- Chapter 6: Ratios and Graphs of Trigonometric Functions
- Chapter 7: Plans and Elevations
- Chapter 8: Loci in Two Dimensions
- Chapter 9: Straight Lines
""",
    "sains": """
OFFICIAL KSSM SCIENCE SYLLABUS INDEX & SCOPE:
FORM 4 SCIENCE:
- Chapter 1: Safety in Laboratory
- Chapter 2: Emergency Aid
- Chapter 3: Techniques of Measuring the Parameters of Body Health
- Chapter 4: Technology of Birth Control
- Chapter 5: Metabolism and Health
- Chapter 6: Support, Locomotion and Growth
- Chapter 7: Body Coordination
- Chapter 8: Elements and Substances
- Chapter 9: Earth
- Chapter 10: Ecosystem
- Chapter 11: Industrial Hygiene
- Chapter 12: Nuclear Energy
- Chapter 13: The Universe
FORM 5 SCIENCE:
- Chapter 1: Microorganisms
- Chapter 2: Nutrition and Food Technology
- Chapter 3: Environmental Sustainability
- Chapter 4: Rate of Reaction
- Chapter 5: Carbon Compounds
- Chapter 6: Electrochemistry
- Chapter 7: Light, Colour and Sight
- Chapter 8: Force and Pressure
- Chapter 9: Space Technology
""",
    "sejarah": """
OFFICIAL KSSM SEJARAH SYLLABUS INDEX & SCOPE:
FORM 4 SEJARAH:
- Bab 1: Warisan Negara Bangsa
- Bab 2: Kebangkitan Nasionalisme
- Bab 3: Konflik Dunia dan Pendudukan Jepun di Negara Kita
- Bab 4: Era Peralihan Kuasa British di Negara Kita
- Bab 5: Persekutuan Tanah Melayu 1948
- Bab 6: Ancaman Komunis dan Pengisytiharan Darurat
- Bab 7: Usaha ke Arah Kemerdekaan
- Bab 8: Pilihan Raya
- Bab 9: Perlembagaan Persekutuan Tanah Melayu 1957
- Bab 10: Pemasyhuran Kemerdekaan
FORM 5 SEJARAH:
- Bab 1: Kedaulatan Negara
- Bab 2: Perlembagaan Persekutuan
- Bab 3: Raja Berperlembagaan dan Demokrasi Berparlimen
- Bab 4: Pentadbiran Negara
- Bab 5: Pembentukan Malaysia
- Bab 6: Cabaran Selepas Pembentukan Malaysia
- Bab 7: Membina Kemakmuran Negara
- Bab 8: Membina Sejahtera Dunia
- Bab 9: Dasar Luar Malaysia
- Bab 10: Kecemerlangan Malaysia di Persada Dunia
""",
    "perniagaan": """
OFFICIAL KSSM PERNIAGAAN SYLLABUS INDEX & SCOPE:
FORM 4 PERNIAGAAN:
- Bab 1: Tujuan Perniagaan dan Pemilikan Perniagaan
- Bab 2: Trend Semasa dalam Perniagaan
- Bab 3: Penetapan Visi, Misi dan Objektif Perniagaan
- Bab 4: Bahagian-bahagian Fungsian Utama dalam Organisasi Perniagaan
FORM 5 PERNIAGAAN:
- Bab 1: Pengurusan Sumber Manusia
- Bab 2: Pengurusan Sumber Fizikal dan Teknologi
- Bab 3: Sumber Pembiayaan Perniagaan
- Bab 4: Penyata Kewangan Perniagaan
""",
    "prinsip perakaunan": """
OFFICIAL KSSM PRINSIP PERAKAUNAN SYLLABUS INDEX & SCOPE:
FORM 4 PERAKAUNAN:
- Bab 1: Pengenalan kepada Perakaunan
- Bab 2: Klasifikasi Akaun dan Persamaan Perakaunan
- Bab 3: Dokumen Perakaunan sebagai Sumber Maklumat
- Bab 4: Buku Catatan Pertama
- Bab 5: Lejar
- Bab 6: Imbangan Duga
- Bab 7: Penyata Kewangan Milikan Tunggal Tanpa Pelarasan
- Bab 8: Pelarasan Tarikh Imbangan dan Penyediaan Penyata Kewangan Milikan Tunggal
- Bab 9: Pembetulan Kesilapan
FORM 5 PERAKAUNAN:
- Bab 1: Analisis dan Tafsiran Penyata Kewangan untuk Membuat Keputusan
- Bab 2: Rekod Tak Lengkap
- Bab 3: Perakaunan untuk Kawalan Dalaman
- Bab 4: Perakaunan untuk Perkongsian
- Bab 5: Perakaunan untuk Syarikat Berhad Menurut Syer
- Bab 6: Perakaunan untuk Kelab dan Persatuan
- Bab 7: Perakaunan Kos
""",
    "ekonomi": """
OFFICIAL KSSM EKONOMI SYLLABUS INDEX & SCOPE:
FORM 4 EKONOMI:
- Bab 1: Pengenalan kepada Ekonomi (Kekurangan, pilihan, kos lepas, sistem ekonomi)
- Bab 2: Pasaran (Permintaan, penawaran, keseimbangan pasaran)
- Bab 3: Wang, Bank dan Pendapatan Individu
- Bab 4: Pengeluaran (Kos, hasil, produktiviti)
FORM 5 EKONOMI:
- Bab 1: Pendapatan Negara
- Bab 2: Dasar Fiskal dan Dasar Monetari
- Bab 3: Inflasi dan Pengangguran
- Bab 4: Perdagangan dan Keantarabangsaan
""",
    "pendidikan islam": """
OFFICIAL KSSM PENDIDIKAN ISLAM SYLLABUS INDEX & SCOPE:
FORM 4 & 5 PENDIDIKAN ISLAM:
- Bidang Al-Quran: Hafazan, Tajwid dan Kefahaman Ayat (Surah pilihan, Larangan Rasuah, Ciri Mukmin Berjaya)
- Bidang Hadis: Menuntut Ilmu, Indahnya Hidup Bersyariat, Kepimpinan yang Adil, Dakwah Islamiah
- Bidang Akidah: Allah Maha Pembalas, Perkara Membatalkan Iman, Hindari Ajaran Sesat, Ahli Sunnah wal Jamaah
- Bidang Fekah: Haji & Umrah, Sembelihan, Korban, Muamalat Islam, Perkahwinan, Pengurusan Harta & Faraid, Jenayah dalam Islam
- Bidang Sirah & Tamadun Islam: Khulafa al-Rasyidin, Kerajaan Umaiyah, Abbasiyah, Uthmaniyyah & Tokoh Islam
- Bidang Akhlak Islamiyah: Adab Terhadap Orang Sakit & OKU, Wasatiyyah, Tawaduk, Istiqamah, Alam Sekitar
""",
    "pendidikan moral": """
OFFICIAL KSSM PENDIDIKAN MORAL SYLLABUS INDEX & SCOPE:
- Bidang 1: Diri Sendiri (Kepercayaan kepada Tuhan, Amanah, Harga Diri, Bertanggungjawab, Berdikari, Kerajinan, Kasih Sayang, Keadilan, Rasional, Kesederhanaan)
- Bidang 2: Keluarga (Hemah Tinggi, Hormat, Kasih Sayang, Tanggungjawab Terhadap Keluarga)
- Bidang 3: Masyarakat (Berdikari, Kerjasama, Toleransi, Kejiranan, Kebebasan)
- Bidang 4: Alam Sekitar (Peka, Menghargai Alam, Keharmonian Antara Manusia dan Alam)
- Bidang 5: Negara (Patriotisme, Hak Asasi Manusia, Demokrasi, Perlembagaan Malaysia, Keamanan)
""",
    "geografi": """
OFFICIAL KSSM GEOGRAFI SYLLABUS INDEX & SCOPE:
- Form 4 & 5: Kemahiran Geografi, Jadual & Graf, Peta Lakar, Penduduk & Petempatan di Malaysia, Pengangkutan, Sumber Semula Jadi, Lembangan Saliran, Isu Alam Sekitar.
""",
    "bahasa melayu": """
OFFICIAL SPM BAHASA MELAYU STRUCTURES:
- Karangan Bahagian A (Bahan Rangsangan) & Karangan Bahagian B (Respon Terbuka)
- Rumusan, Pemahaman Petikan, Novel Komsas, Tatabahasa, Morfologi, Sintaksis & Peribahasa.
""",
    "bahasa inggeris": """
OFFICIAL SPM CEFR ENGLISH WRITING STRUCTURES:
- Part 1: Short Communicative Message, Part 2: Guided Writing, Part 3: Extended Essay / Article / Review / Story (Band 5 Standard).
- Reading Comprehension, Use of English, Listening, Speaking & Literature.
""",
    "pjk": """
OFFICIAL KSSM PJK (PENDIDIKAN JASMANI & KESIHATAN):
- Kecergasan Fizikal, Kemahiran Sukan & Permainan, Kesihatan Diri & Keluarga, Kesukanan & Keselamatan.
""",
    "ask": """
OFFICIAL KSSM ASAS KARSIN MATIK (ASK) / COMPUTER SCIENCE:
- Konsep Asas Perkomputeran, Perwakilan Data, Algoritma, Kod Arahan, Pangkalan Data (SQL), Reka Bentuk Web (HTML/CSS/JS).
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
    data = request.json
    user_prompt = data.get("prompt", "")
    task_id = data.get(
        "task", "create_notes" if "notes" in user_prompt.lower() else "explain"
    )
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
            model_to_use = "qwen/qwen3.8-27b"  # Active vision model
        else:
            messages.append({"role": "user", "content": user_prompt})
            model_to_use = "openai/gpt-oss-120b"  # Active text model

        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
        )
        return jsonify(
            {"success": True, "task": task_id, "reply": response.choices[0].message.content}
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
