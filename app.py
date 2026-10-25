

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


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}
        user_prompt = data.get("message", "") or data.get("prompt", "")
        selected_subject = data.get("subject", "Biologi")
        output_language = data.get("language", "English")

        system_prompt = (
            f"You are Core Study AI v3.0, an elite Malaysian SPM tutor for {selected_subject}. "
            f"Answer clearly in {output_language} based strictly on the KSSM syllabus and exam marking standards."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        
        return jsonify({"success": True, "reply": response.choices[0].message.content})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
