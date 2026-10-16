import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Assign labels
fake["label"] = 0
real["label"] = 1

# Combine the datasets
data = pd.concat([fake, real], ignore_index=True)
data = data.dropna(subset=["text"])
data = data.drop_duplicates(subset=["text"])

# Combine headline and article text
data["content"] = (
    data["title"].fillna("") + " " + data["text"]
)

X = data["content"]
y = data["label"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build the NLP and ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english", max_features=50000
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
print("Training model... Please wait.")
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model accuracy:", round(accuracy * 100, 2), "%")
print(classification_report(y_test, predictions))

# Save the trained model
joblib.dump(model, "fake_news_model.pkl")
print("Model saved successfully!")
 