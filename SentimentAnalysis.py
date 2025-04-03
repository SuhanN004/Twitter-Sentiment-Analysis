from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Download stopwords once
nltk.download('stopwords')

# Load the dataset
file_path = r"twitter_training.csv"
data = pd.read_csv(file_path , header = None ,names=['number' , 'Border' , 'label' , 'message']) # Adjusting the column names
data = data[['label', 'message']].dropna()


# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)

# Clean the text column
data['clean_text'] = data['message'].apply(clean_text)


data['label'] = data['label'].str.lower()
print(data['label'].value_counts())
print("Unique sentiment categories:")
print(data['label'].unique())
# Step 2: Remove 'irrelevant'
data = data[~data['label'].isin(['irrelevant'])]
# Map sentiment to numbers
data['label'] = data['label'].map({'positive': 1, 'neutral': 1, 'negative': -1})

# Drop rows with NaN labels
data = data.dropna(subset=['label'])

# Count each category
print("Label distribution:")
print(data['label'].value_counts())
print("Unique sentiment categories:")
print(data['label'].unique())


# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['clean_text'])
y = data['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot with TP/TN/FP/FN in mind
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    xticklabels=['Predicted: Negative', 'Predicted: Positive'],
    yticklabels=['Actual: Negative', 'Actual: Positive']
)
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.title("Confusion Matrix: TP, TN, FP, FN View")
plt.show()


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Sample test messages
sample_messages = [
    "I love this product! Totally worth it.",
    "This is the worst service I’ve ever experienced.",
    "Not bad, could be better.",
    "What is this even supposed to be?",
    "Absolutely amazing experience!"
]

# Preprocess sample messages
clean_samples = [clean_text(msg) for msg in sample_messages]

# Convert to vector using the same TF-IDF vectorizer
sample_vectors = vectorizer.transform(clean_samples)

# Predict using trained model
sample_preds = model.predict(sample_vectors)

# Display results
for msg, pred in zip(sample_messages, sample_preds):
    label = {1: "Positive", -1: "Negative"}.get(pred, "Unknown")
    print(f"Message: {msg}\nPredicted Sentiment: {label}\n")


sample_tweets = [
    "Love this phone 😍🔥",                          # emoji
    "Ths prduct is amazng!",                        # misspelling
    "Oh great, another update that broke everything",  # sarcasm
    "Bad",                                           # very short
    "Good",
    "This product is absolutely, unequivocally, the worst and most disappointing experience of my life so far."  # long + negative
]
# Preprocess
cleaned_samples = [clean_text(msg) for msg in sample_tweets]

# Vectorize
sample_vecs = vectorizer.transform(cleaned_samples)

# Predict
sample_preds = model.predict(sample_vecs)

# Display results using consistent label map
for msg, pred in zip(sample_tweets, sample_preds):
    label = {1: "Positive", -1: "Negative"}.get(pred, "Unknown")
    print(f"Message: {msg}\nPredicted Sentiment: {label}\n")


sample_noisy = [
    "Thisss phonee is amaaazing!!!",     # repeated characters
    "I l0ve thiss!!",                    # number used for letter
    "The update...ughhh broke it ag@in", # random symbols
    "!!!",
    "???"
]

# Preprocess
cleaned_samples = [clean_text(msg) for msg in sample_noisy]

# Vectorize
sample_vecs = vectorizer.transform(cleaned_samples)

# Predict
sample_preds = model.predict(sample_vecs)

# Display results using consistent label map
for msg, pred in zip(sample_noisy, sample_preds):
    label = {1: "Positive", -1: "Negative"}.get(pred, "Unknown")
    print(f"Message: {msg}\nPredicted Sentiment: {label}\n")


