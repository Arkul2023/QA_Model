from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)

# Load model and tokenizer from saved directory
model_path = "saved_model"  
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

@app.route('/')
def home():
    """Renders the HTML form for user input."""
    return render_template("QA.html")

@app.route('/qa', methods=['POST'])
def answer_question():
    """Handles form submission & API calls for QA."""
    if request.method == "POST":
        question = request.form.get("question")  # Get question from HTML form
        context = request.form.get("context")  # Get context from HTML form
        
        if not question or not context:
            return render_template("QA.html", error="Both fields are required!")

        # Process with model
        result = qa_pipeline(question=question, context=context)

        return render_template("QA.html", question=question, context=context, answer=result["answer"])

if __name__ == '__main__':
    app.run(debug=True)
