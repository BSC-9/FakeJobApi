from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

app = Flask(__name__)
CORS(app)

# Path where you uploaded your model (modify if needed)
model_path = "bc0985/Fake_Job_LLM"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Define the pipeline globally
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

# Function to classify job listings
def run_inference(job_listing):
    instruction = "Classify whether the following job listing is real or fake. Provide your reasoning."
    prompt = f"{instruction}\n\n{job_listing}\n\nAnswer:"

    output = pipe(prompt)
    generated_text = output[0]["generated_text"]

    if "Answer:" in generated_text:
        return generated_text.split("Answer:")[-1].strip()
    else:
        return generated_text.strip()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    job_listing = data.get("job_listing", "")
    prediction = run_inference(job_listing)
    return jsonify({"Prediction": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
