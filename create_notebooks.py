import nbformat as nbf
import os

# Helper to create a notebook
def create_notebook(filename, cells_data):
    nb = nbf.v4.new_notebook()
    cells = []
    for cell_type, content in cells_data:
        if cell_type == 'markdown':
            cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == 'code':
            cells.append(nbf.v4.new_code_cell(content))
    nb['cells'] = cells
    with open(filename, 'w') as f:
        nbf.write(nb, f)

# 01_EDA
eda_cells = [
    ('markdown', '# Exploratory Data Analysis\n\nLoading the dataset and checking its properties.'),
    ('code', '''import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load datasets
fake_df = pd.read_csv('../data/Fake.csv')
true_df = pd.read_csv('../data/True.csv')

# Assign labels (1 for Fake, 0 for True)
fake_df['label'] = 1
true_df['label'] = 0

# Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save combined raw data
df.to_csv('../data/raw.csv', index=False)
df.head()
'''),
    ('markdown', '## Class Distribution\n\nLet\'s see how balanced the dataset is.'),
    ('code', '''plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='label')
plt.title('Class Distribution (0 = True, 1 = Fake)')
plt.show()
'''),
    ('markdown', '## Text Length Distribution\n\nComparing the length of fake vs true news articles.'),
    ('code', '''df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10, 6))
sns.histplot(df[df['label'] == 0]['text_length'], color='blue', label='True', bins=50, alpha=0.5)
sns.histplot(df[df['label'] == 1]['text_length'], color='red', label='Fake', bins=50, alpha=0.5)
plt.xlim(0, 2000) # limit to 2000 words for better visualization
plt.title('Text Length Distribution')
plt.legend()
plt.show()
'''),
    ('markdown', '## Word Clouds\n\nMost frequent words in True vs Fake news.'),
    ('code', '''true_text = " ".join(df[df['label'] == 0]['text'].astype(str))
fake_text = " ".join(df[df['label'] == 1]['text'].astype(str))

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
wc_true = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(true_text)
plt.imshow(wc_true, interpolation='bilinear')
plt.title('True News Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
wc_fake = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(fake_text)
plt.imshow(wc_fake, interpolation='bilinear')
plt.title('Fake News Word Cloud')
plt.axis('off')

plt.show()
''')
]

# 02_preprocessing
prep_cells = [
    ('markdown', '# Text Preprocessing\n\nCleaning text by removing URLs, HTML, punctuation, stopwords, and applying lemmatization.'),
    ('code', '''import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Load raw combined dataset
df = pd.read_csv('../data/raw.csv')

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>', '', text)          # Remove HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # Keep only letters
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)
'''),
    ('markdown', 'Applying preprocessing. This might take a while on 40k+ articles.'),
    ('code', '''# Apply to a sample first if you want to test, here we do full data
# We combine title and text for a richer feature
df['full_text'] = df['title'] + " " + df['text']
print("Preprocessing started...")
df['processed_text'] = df['full_text'].apply(preprocess)
print("Preprocessing finished!")

# Drop any rows where processed_text became empty
df = df[df['processed_text'].str.strip() != '']

# Save to processed.csv
df.to_csv('../data/processed.csv', index=False)
df[['full_text', 'processed_text', 'label']].head()
''')
]

# 03_baseline
baseline_cells = [
    ('markdown', '# Baseline Models (TF-IDF)\n\nTraining Logistic Regression and Random Forest models as baselines.'),
    ('code', '''import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
df = pd.read_csv('../data/processed.csv')

# Drop NA rows just in case
df = df.dropna(subset=['processed_text', 'label'])
'''),
    ('markdown', '## Vectorization and Splitting'),
    ('code', '''# Use TF-IDF vectorizer (max 5000 features for efficiency)
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['processed_text'])
y = df['label']

# 85/15 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
'''),
    ('markdown', '## Logistic Regression Model'),
    ('code', '''lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print("Logistic Regression Classification Report:")
print(classification_report(y_test, lr_preds))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, lr_preds), annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
'''),
    ('markdown', '## Random Forest Model'),
    ('code', '''rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, rf_preds))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Greens')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
''')
]

if not os.path.exists('notebooks'):
    os.makedirs('notebooks')

create_notebook('notebooks/01_EDA.ipynb', eda_cells)
create_notebook('notebooks/02_preprocessing.ipynb', prep_cells)
create_notebook('notebooks/03_baseline.ipynb', baseline_cells)
print("Notebooks created successfully.")
