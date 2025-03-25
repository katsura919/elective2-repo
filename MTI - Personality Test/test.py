import joblib
import re
import numpy as np

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Load the saved model and vectorizer
model = joblib.load("mbti_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_to_type = {
    0: "ENTP", 1: "ENTJ", 2: "ENFP", 3: "ENFJ", 4: "ESTP", 5: "ESTJ",
    6: "ESFP", 7: "ESFJ", 8: "INTP", 9: "INTJ", 10: "INFP", 11: "INFJ",
    12: "ISTP", 13: "ISTJ", 14: "ISFP", 15: "ISFJ"
}

print("Model and vectorizer loaded!")

def predict_mbti(text):
    text_cleaned = clean_text(text)
    text_vectorized = vectorizer.transform([text_cleaned])
    
    # Get prediction probabilities
    probabilities = model.predict_proba(text_vectorized)[0]  # Returns an array of probabilities
    label = np.argmax(probabilities)  # Get the most probable class
    confidence = probabilities[label]  # Get the probability of the predicted class
    
    return label, label_to_type[label], confidence

if __name__ == "__main__":
    print("Enter text to predict MBTI type:")
    user_input = input()
    label, predicted_type, confidence = predict_mbti(user_input)
    print(f"Predicted MBTI Type: {predicted_type} (Class {label})")
    print(f"Confidence Level: {confidence:.4f}")
