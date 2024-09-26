import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Load Dataset
df = pd.read_csv("New folder\spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ['label', 'message']

# Encode label column
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Feature extraction using Bag of Words
cv = CountVectorizer(stop_words='english')
X = cv.fit_transform(df['message'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save model and CountVectorizer using pickle
with open("spam_detection_model.pkl", "wb") as model_file:
    pickle.dump((model, cv), model_file)