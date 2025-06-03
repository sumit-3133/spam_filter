import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
# You might need to import the exact model class you used (e.g., LogisticRegression)
# if your model relies on specific classes for loading
from sklearn.linear_model import LogisticRegression # Or SVC, RandomForestClassifier, etc.

st.set_page_config(page_title="Spam Detector", page_icon="✉️")

# --- Load the pre-trained model and vectorizer ---
@st.cache_resource # Cache the loading to prevent reloading on every rerun
def load_resources():
    try:
        with open('feature_extractor.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    except FileNotFoundError:
        st.error("Error: Model or vectorizer file not found. Make sure 'tfidf_vectorizer.pkl' and 'spam_model.pkl' are in the same directory.")
        st.stop() # Stop the app if files are missing

feature_extraction, model = load_resources()


# --- Streamlit App ---


st.title("✉️ SMS Spam Detector")
st.write("Enter an SMS message below to check if it's spam or not.")

# Text input for the user
user_input = st.text_area("Enter SMS Message", height=150, placeholder="Type your message here...")

if st.button("Predict"):
    if user_input:
        # Transform the input using the loaded TfidfVectorizer
        input_mail_extraction = feature_extraction.transform([user_input])

        # Make the prediction
        # The actual prediction logic from your original code
        y_predict = model.predict(input_mail_extraction)

        if y_predict == 0:
            st.success("🎉 Not a spam!")
        else:
            st.error("🚨 Spam!")
    else:
        st.warning("Please enter a message to predict.")

st.markdown("---")
st.markdown("Developed by Sumit Kumar")