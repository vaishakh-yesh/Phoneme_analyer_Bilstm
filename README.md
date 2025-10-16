# Phoneme_analyer_Bilstm

# Save the model to a file
model.save('bilstm_model.h5')
print("Model saved successfully.")


import joblib

# Save the label encoder to a file
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Label encoder saved successfully.")


import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import librosa  # Import librosa here

# Function to extract MFCC features (copied from previous cell)
def extract_features(audio_path, sr=16000, duration=5, n_mfcc=13):
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)

        # If audio is shorter than the desired duration, pad it with zeros
        if len(y) < sr * duration:
            padding = sr * duration - len(y)
            y = np.pad(y, (0, padding))

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Return the mean of MFCC features over time (for dimensionality reduction)
        return np.mean(mfccs.T, axis=0)

    except Exception as e:
        print(f"Error encountered while parsing file: {audio_path}, Error: {e}")
        return None

# Function to make predictions
def predict_audio_class(audio_path, model, label_encoder, sr=16000, duration=5, n_mfcc=13):
    # Extract features from the audio file
    features = extract_features(audio_path, sr=sr, duration=duration, n_mfcc=n_mfcc)

    if features is not None:
        # Reshape features to match model input shape
        features = features.reshape(1, 1, 13)  # (1 sample, 1 time step, 13 features)

        # Predict the class probabilities
        predictions = model.predict(features)

        # Get the index of the maximum probability
        predicted_class_index = np.argmax(predictions)

        # Decode the class index to a label
        predicted_label = label_encoder.inverse_transform([predicted_class_index])

        return predicted_label[0]
    else:
        return None

# Load your trained model
# (Replace 'bilstm_model.h5' with the path to your saved model file)
model = load_model('bilstm_model.h5')

# Load the label encoder (ensure you save it using joblib or pickle during training)
import joblib
label_encoder = joblib.load('label_encoder.pkl')  # Replace with your saved file path

# Path to new audio file
new_audio_path = '/content/MAL 13.wav'

# Predict the class
predicted_label = predict_audio_class(new_audio_path, model, label_encoder)

if predicted_label:
    print(f"The predicted label for the audio is: {predicted_label}")
else:
    print("Could not process the audio file.")





  warnings.warn(
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 593ms/step
The predicted label for the audio is: malayalam  
