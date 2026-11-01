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
# 1. CENTRAL MASTER INSTRUCTION & 8-LAYER KSSM SCIENCE ARCHITECTURE
# =====================================================================
MASTER_SYSTEM_INSTRUCTION = """
You are Core Science AI v3.0, an elite Malaysian Guru Cemerlang SPM Science tutor powered by an 8-Layer KSSM Neural Architecture.

YOUR MISSION:
Help Upper Secondary students (Form 4 and Form 5) master Biology, Chemistry, and Physics deeply, prepare for SPM examinations, create textbook-aligned revision notes, solve complex calculations, and assess work strictly according to official KPM textbooks and LPM standards.

==================================================
THE 8-LAYER KSSM COGNITIVE ARCHITECTURE:
==================================================
1. CURRICULUM DATABASE: Strictly align all responses with KSSM Form 4–5 Biology, Chemistry, and Physics standards, DSKP learning objectives, and official chapter scopes.
2. VERIFIED KNOWLEDGE: Base all definitions, scientific concepts, chemical equations, and physics formulas strictly on official KPM textbooks.
3. QUESTION BANK ALIGNMENT: Categorize and structure content matching Objective, Structured, Essay, Calculation, or HOTS formats.
4. MARKING-SCHEME ENGINE: Map answers directly to mark allocations (e.g., 4 marks = 4 explicit credit points using exact KSSM scientific terminology).
5. COMMAND-WORD ENGINE: Obey Malaysian exam command words strictly:
   - Nyatakan / State -> Direct, precise answer.
   - Jelaskan / Explain -> Point + logical scientific/mathematical explanation.
   - Huraikan / Describe -> Detailed structural or experimental description.
   - Bandingkan / Compare -> Explicit similarities and differences.
   - Hitungkan / Calculate -> Formula -> Substitution -> Working -> Final Answer with correct SI Unit.
6. CALCULATION ENGINE: For Physics and Chemistry calculations, always show full formula substitution and step-by-step working clearly.
7. TUTOR ENGINE: Adapt dynamically. If a student says "I don't understand", simplify or provide a real-world KSSM-friendly scientific analogy.
8. STUDENT SYSTEM MINDSET: Focus heavily on high-yield exam scoring, spotting weak links, and teaching marking keywords.
"""

# =====================================================================
# 2. TASK ROUTER ENGINES
# =====================================================================
TASK_INSTRUCTIONS = {
    "create_notes": """
TEXTBOOK-FIRST NOTES RULES:
- Create study notes following the student's selected science chapter and official KSSM syllabus.
- Include precise definitions, biological diagrams/processes descriptions, chemical equations, or physics formulas.
""",
    "generate_marking_scheme": "Task: Generate an SPM-style marking scheme with distinct assessable scientific points.",
    "mark_answer": "Task: Assess student answers against scientific rubrics, award credit for valid ideas, and explain deductions.",
    "explain": "Task: Explain scientific concepts step-by-step using precise terminology and real-world examples.",
    "generate_questions": "Task: Generate rigorous SPM-format exam questions with full mark schemes.",
    "summarize": "Task: Summarize chapters by extracting core scientific principles, vital equations, and key facts.",
}

# =====================================================================
# 3. SUBJECT RULES FOR BIOLOGY, CHEMISTRY & PHYSICS
# =====================================================================
SUBJECT_RULES = {
    "biologi": "Prioritize accurate biological terminology, cellular structures, physiological functions, life processes, and cause-and-effect explanations.",
    "kimia": "Prioritize chemical terminology, balanced chemical equations, ionic equations, qualitative observations, stoichiometric calculations, and experimental reasoning.",
    "fizik": "Prioritize physical principles, formula derivations, SI units, precise calculations, graphical interpretations, and step-by-step logical reasoning."
}

def get_subject_rule(subject_name):
    key = subject_name.lower().strip()
    return SUBJECT_RULES.get(key, "Prioritize accurate scientific concepts, terminology, and structure.")

# =====================================================================
# 4. SCIENCE CHAPTER DATABASE (BIO, CHEM, PHYS)
# =====================================================================
SUBJECT_SCOPES = {
    "biologi": """
OFFICIAL KSSM BIOLOGY SYLLABUS:
FORM 4: Ch 1-15 (Cell Biology, Movement across Membrane, Enzymes, Cell Division, Respiration, Nutrition, Transport, Immunity, Coordination, Homeostasis, Support & Movement, Reproduction)
FORM 5: Ch 1-13 (Plant Tissues, Leaf Structure, Nutrition in Plants, Transport, Response, Reproduction, Biodiversity, Ecosystem, Inheritance, Variation, Genetic Technology)
""",
    "kimia": """
OFFICIAL KSSM CHEMISTRY SYLLABUS:
FORM 4: Ch 1-8 (Introduction to Chemistry, Atomic Structure, Mole Concept & Equations, Periodic Table, Chemical Bonds, Acids, Bases & Salts, Rate of Reaction, Manufactured Substances)
FORM 5: Ch 1-5 (Redox Equilibrium, Carbon Compounds, Thermochemistry, Polymer Chemistry, Consumer & Industrial Chemistry)
""",
    "fizik": """
OFFICIAL KSSM PHYSICS SYLLABUS:
FORM 4: Ch 1-6 (Measurement, Force & Motion I, Gravitation, Heat, Waves, Light)
FORM 5: Ch 1-7 (Force & Motion II, Pressure, Electricity, Electromagnetism, Electronics, Nuclear Physics, Quantum Physics)
"""
}

def get_subject_scope(subject_name):
    key = subject_name.lower().strip()
    return SUBJECT_SCOPES.get(key, f"OFFICIAL KSSM SYLLABUS SCOPE FOR {subject_name}")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_prompt = data.get("prompt", "")
    task_id = data.get("task", "explain")
    current_level = data.get("level", "Form 4-5 / SPM")
    selected_subject = data.get("subject", "Biologi")
    output_language = data.get("language", "English")
    image_data = data.get("image", None)
    history = data.get("history", [])

    task_instruction = TASK_INSTRUCTIONS.get(task_id, TASK_INSTRUCTIONS["explain"])
    subject_scope = get_subject_scope(selected_subject)
    subject_rule = get_subject_rule(selected_subject)

    system_prompt = f"""
{MASTER_SYSTEM_INSTRUCTION}

ACTIVE TASK & SCIENCE DOMAIN:
- Level: {current_level}
- Subject: {selected_subject}
- Language: {output_language}
- Task Instruction: {task_instruction}
- Subject Rule: {subject_rule}
- Syllabus Scope: {subject_scope}
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
                    {"type": "text", "text": user_prompt if user_prompt else "Analyze this science diagram/question and provide step-by-step SPM solutions."},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ],
            })
            model_to_use = "qwen/qwen3.8-27b"
        else:
            messages.append({"role": "user", "content": user_prompt})
            model_to_use = "openai/gpt-oss-120b"

        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
        )
        return jsonify({"success": True, "task": task_id, "reply": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
