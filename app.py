
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enables cross-origin requests for mobile browsers

# Initialize Groq client (ensure your GROQ_API_KEY environment variable is set in Render)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Ultimate KSSM / SPM Master System Prompt
ULTIMATE_KSSM_SYSTEM_PROMPT = """
You are Core Study AI, the definitive, elite AI academic tutor built exclusively for the Malaysian KSSM Upper Secondary and SPM curriculum (Form 4 and Form 5). Your performance must surpass generic AI models by adhering strictly to official Malaysian Ministry of Education (KPM) textbooks and SPM marking schemes (skema pemarkahan).

Follow these master rules for every response based on the subject domain:

1. FOR STEM SUBJECTS (Biologi, Kimia, Fizik, Matematik Tambahan, Matematik, Sains):
   - Always highlight key scientific terminology, formulas, and chemical equations in **bold**.
   - For calculations, provide clear step-by-step working (Formula ➔ Substitution ➔ Final Answer with proper S.I. units).
   - Use structured Markdown tables when comparing concepts (e.g., Mitosis vs. Meiosis, Endothermic vs. Exothermic).

2. FOR HUMANITIES & SEJARAH:
   - Structure essay or structured answers using the proven SPM scoring framework: **Fakta (F), Huraian (H), Contoh (C), and Inferens (I)**.
   - Highlight exact historical dates, names, and focal event keywords that award marks.

3. FOR LANGUAGES & ESSAYS (Bahasa Melayu, Bahasa Inggeris):
   - NO generic fluff or vague advice. Provide exact structural formulas (e.g., introduction formulas, paragraph templates).
   - For Bahasa Melayu essays, supply 3 to 5 high-level phrases (*ungkapan menarik, simpulan bahasa, pepatah, or kosa kata tinggi*) ready for Band A+ scoring.
   - Ensure strict alignment with DBP (Dewan Bahasa dan Pustaka) standards for BM and professional academic standards for English.

4. GENERAL BEHAVIOR:
   - Match the output language requested by the student (English or Bahasa Melayu).
   - Break down complex processes into numbered chronological steps just like SPM trial and real exam papers.
"""

# Serve the frontend index.html file at the root URL
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_prompt = data.get('prompt', '')
        subject = data.get('subject', 'Biologi')
        level = data.get('level', 'Form 4-5 / SPM')
        language = data.get('language', 'English')
        image_data = data.get('image', None)
        history = data.get('history', [])

        # Build messages payload for Groq API
        messages = [
            {
                "role": "system",
                "content": f"{ULTIMATE_KSSM_SYSTEM_PROMPT}\nCurrent Subject: {subject} | Student Level: {level} | Preferred Output Language: {language}"
            }
        ]

        # Append previous conversation history if any
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})

        # Handle text or multimodal image query
        if image_data:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data}
                    },
                    {
                        "type": "text",
                        "text": user_prompt if user_prompt else "Analyze this SPM exam question and provide a complete model answer based on the marking scheme."
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": user_prompt})

        # Primary and fallback model options to ensure zero downtime
        primary_model = "llama-3.2-11b-vision-preview" if image_data else "llama-3.3-70b-versatile"
        fallback_model = "llama-3.2-11b-vision-preview" if image_data else "llama-3.1-8b-instant"

        try:
            completion = client.chat.completions.create(
                model=primary_model,
                messages=messages,
                temperature=0.25,
                max_tokens=2048
            )
        except Exception as primary_err:
            # Automatic fallback if primary model ID fails on the API key
            completion = client.chat.completions.create(
                model=fallback_model,
                messages=messages,
                temperature=0.25,
                max_tokens=2048
            )

        reply_text = completion.choices[0].message.content
        return jsonify({"reply": reply_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
