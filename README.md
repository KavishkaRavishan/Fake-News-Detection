# Fake News Detection Project

This project aims to detect fake news using AI techniques including NLP preprocessing, TF-IDF baseline models, BERT fine-tuning, and LLM Prompt Engineering.

## Project Structure

- `data/`: Place your datasets here (e.g., ISOT Fake News Dataset).
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
- `src/`: Reusable Python modules and helper scripts.
- `demo/`: Gradio web application for model demonstration.

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK data:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## Usage

1. Start with the notebooks in the `notebooks/` directory sequentially.
2. After training the BERT model in `04_bert.ipynb`, run the demo app:
   ```bash
   python demo/app.py
   ```
