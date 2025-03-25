import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the saved model
loaded_model = tf.keras.models.load_model('LSTM_Sentiment_analysis.h5')

def Predict(text, model=loaded_model, tokenizer=tokenizer, max_length=10):
    """
    Preprocesses input text, predicts sentiment using the trained model, 
    and returns the predicted class and probability.
    """
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    
    # Get model prediction
    prob = model.predict(padded_sequence)[0][0]
    
    # Convert probability to class (assuming binary classification: 0 = Negative, 1 = Positive)
    pred_class = 1 if prob >= 0.5 else 0
    
    return pred_class, prob

# Run manually in the terminal
if __name__ == "__main__":
    new_text = input("Enter a sentence to analyze sentiment: ")  # Take input from the user
    pred_class, prob = Predict(new_text)
    
    sentiment = "Positive" if pred_class == 1 else "Negative"
    print(f"\nClass Prediction: {sentiment} ({pred_class}) with Probability: {prob:.4f}")
