import os
import google.generativeai as genai
import google.generativeai.types as types
from flask import Flask, request, jsonify

# Load API key securely
API_KEY = "AIzaSyBrYh2jmIjEKJAhtaZdWxi0CGhv1Ed4E50"  # Replace with your actual API key
genai.configure(api_key=API_KEY)  # ✅ Correct way to set API key

# Initialize model
model = genai.GenerativeModel("gemini-2.0-flash")

# Initialize Flask app
app = Flask(__name__)

def generate_response(input_text):
    """Generates a response to verify if a job is fake or genuine."""
    
    input_text += " Determine if the job information is fake or genuine in percentages with a one-line explanation. Keep the response under 25 words."

    response = model.generate_content(
        input_text,
        generation_config=types.GenerationConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=100,
            response_mime_type="text/plain",
        ),
    )

    return response.text.strip().replace('•', '*')  # Clean response

@app.route("/check-job", methods=["POST"])
def check_job():
    """API endpoint to check if a job is fake or genuine."""
    data = request.json
    job_details = data.get("job_details", "")

    if not job_details:
        return jsonify({"error": "Job details are required"}), 400

    result = generate_response(job_details)
    return jsonify({"result": result})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
