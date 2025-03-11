import os
from flask import Flask, request, jsonify
import google.generativeai as genai
import google.generativeai.types as types

API_KEY = "AIzaSyBrYh2jmIjEKJAhtaZdWxi0CGhv1Ed4E50"

client = genai.Client(api_key=API_KEY)
model = "gemini-2.0-flash"  # Using Gemini 2.0 Flash for fast responses

app = Flask(__name__)

def generate_response(input_text):
    """Generates a response to verify if a job is fake or genuine."""
    
    input_text += " Determine if the job information is fake or genuine in percentages with a one-line explanation. Keep the response under 25 words."

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=input_text)],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=100,  # Limit response size
        response_mime_type="text/plain",
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    return response.text.strip().replace('•', '*')

@app.route("/check-job", methods=["POST"])
def check_job():
    """API endpoint to check if a job is fake or genuine."""
    data = request.json
    job_details = data.get("job_details", "")

    if not job_details:
        return jsonify({"error": "Job details are required"}), 400

    result = generate_response(job_details)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
