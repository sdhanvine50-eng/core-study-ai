
from flask import Flask, request, jsonify
import google.generativeai as genai
import os

app = Flask(__name__)

# Configure Gemini API Key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY"))

# Command Prompt Definitions for SPM/KSSM
COMMAND_PROMPTS = {
    "/explain": "Break down this topic using the Feynman technique so a Form 4/5 student can easily understand it, avoiding overly dense jargon.",
    "/analogy": "Provide a creative, real-world analogy to help visualize this scientific process (e.g., comparing mitosis to a factory assembly line or copying machine).",
    "/bilingual": "Explain this key term in both English and Bahasa Malaysia, highlighting the exact SPM keyword required for marking schemes.",
    "/spm_drill": "Generate 2 SPM-style structured questions (Paper 2 format) based on this topic, complete with a model answer and marking scheme breakdown.",
    "/keywords": "Extract and list all the crucial KSSM keywords from the current notes that examiners look for in grading.",
    "/mnemonic": "Create a catchy mnemonic or memory trick to help remember the steps of this process in the correct order.",
    "/quiz": "Test my knowledge with 3 multiple-choice questions based on these notes. Wait for my answer before giving the feedback and explanation.",
    "/summary": "Condense the current lesson text into 5 high-yield bullet points for quick last-minute revision.",
    "/flashcard": "Turn the key definitions on this page into a Q&A flashcard format (Front: Question/Term, Back: Answer/Definition)."
}

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_prompt = data.get('prompt', '').strip()
        subject = data.get('subject', 'General')
        level = data.get('level', 'Form 4-5 / SPM')
        language = data.get('language', 'English')
        image_data = data.get('image', None)
        chat_history = data.get('history', [])

        # Check for matching command prefix
        command_instruction = ""
        matched_command = None
        for cmd, prompt_template in COMMAND_PROMPTS.items():
            if user_prompt.lower().startswith(cmd):
                matched_command = cmd
                command_instruction = prompt_template
                # Extract topic after the command if present
                clean_topic = user_prompt[len(cmd):].strip()
                if clean_topic:
                    user_prompt = clean_topic
                break

        # System context construction
        system_instruction = (
            f"You are Core Study AI, an expert tutor for Malaysian Upper Secondary students ({level}) studying {subject}. "
            f"Respond primarily in {language}. Ensure alignment with the Malaysian KSSM curriculum and SPM marking schemes.\n\n"
        )

        if command_instruction:
            system_instruction += f"USER INSTRUCTION MODE [{matched_command}]: {command_instruction}\n"

        full_prompt = f"{system_instruction}\nStudent Question/Topic: {user_prompt}"

        # Initialize Gemini Model
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Handle image input if provided
        if image_data:
            import base64
            header, encoded = image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            image_parts = [{"mime_type": "image/jpeg", "data": image_bytes}]
            response = model.generate_content([full_prompt, image_parts[0]])
        else:
            response = model.generate_content(full_prompt)

        return jsonify({"reply": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
