from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

app = Flask(__name__)

# Hugging Face model path
MODEL_NAME = "bc0985/Fake_Job_LLM"

# Load model and tokenizer from the "check" folder
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder="check")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    subfolder="check",  # Specify the subfolder where the model is stored
    device_map="auto",
    torch_dtype=torch.float16  # Use float16 for lower memory usage
)

# Define the text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.2,
)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    job_listing = data.get("job_listing", "")

    if not job_listing:
        return jsonify({"error": "No job listing provided"}), 400

    # Construct the prompt
    instruction = "Classify whether the following job listing is real or fake. Provide your reasoning."
    prompt = f"{instruction}\n\n{job_listing}\n\nAnswer:"

    # Run inference
    output = pipe(prompt)
    response_text = output[0]["generated_text"].split("Answer:")[-1].strip()

    return jsonify({"prediction": response_text})

@app.route("/", methods=["GET"])
def home():
    return "Flask API for Job Classification is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
