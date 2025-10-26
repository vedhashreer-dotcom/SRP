import streamlit as st
import librosa
import librosa.display
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

#Load and save XGBoost
model_path = r"C:\Vedha\miniproject\xgb_model.pkl"

if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Make sure the pickle exists!")
else:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    st.title("Heart Sound Classification")
    st.write("Upload a PCG (.wav) file and the model will predict Normal or Abnormal.")

    #Upload file
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

    if uploaded_file is not None:
        try:
            # Load audio
            signal, sr = librosa.load(uploaded_file, sr=None)

            #Extract MFCC features
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
            feat = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
            feat = feat.reshape(1, -1)

            #Predict
            pred = model.predict(feat)[0]
            pred_proba = model.predict_proba(feat)[0]

            result = "Abnormal" if pred == 1 else "Normal"
            st.success(f"Prediction: {result}")

            # Show prediction probability
            st.subheader("Prediction Probabilities")
            st.bar_chart({"Normal": pred_proba[0], "Abnormal": pred_proba[1]})

            #MFCC Visualization
            st.subheader("MFCC Visualization")
            fig, ax = plt.subplots(figsize=(8, 4))
            img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=ax)
            fig.colorbar(img, ax=ax)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing file: {e}")
