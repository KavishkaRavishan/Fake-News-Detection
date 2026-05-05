import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load your saved model
# Make sure to handle the case where the model isn't trained yet
try:
    tokenizer = BertTokenizer.from_pretrained('./saved_model')
    model = BertForSequenceClassification.from_pretrained('./saved_model')
    model_loaded = True
except Exception as e:
    print(f"Warning: Model not found at ./saved_model. Please train the model first. Error: {e}")
    model_loaded = False

def predict(text):
    if not model_loaded:
        return "Error: Model not trained yet. Please run the BERT notebook first."
        
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    fake_prob = probs[0][0].item()
    real_prob = probs[0][1].item()
    label = "REAL" if real_prob > fake_prob else "FAKE"
    confidence = max(real_prob, fake_prob) * 100
    return f"{label} (Confidence: {confidence:.1f}%)"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=10, placeholder="Paste a news article here..."),
    outputs=gr.Textbox(label="Prediction"),
    title="Fake News Detector",
    description="Paste a news article to check if it is real or fake. Powered by fine-tuned BERT.",
    examples=[
        ["Scientists have discovered a new species of dinosaur in Argentina..."],
        ["BREAKING: Government confirms aliens exist, cover-up exposed..."],
    ]
)

if __name__ == "__main__":
    demo.launch()
