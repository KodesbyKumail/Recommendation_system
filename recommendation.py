import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import speech_recognition as sr
import os

# --- Load and preprocess data directly ---
styles_df = pd.read_csv('styles.csv', on_bad_lines='warn')

# Create combined features for vectorization
styles_df['combined_features'] = (
        styles_df['gender'].fillna('') + " " +
        styles_df['masterCategory'].fillna('') + " " +
        styles_df['subCategory'].fillna('') + " " +
        styles_df['articleType'].fillna('') + " " +
        styles_df['baseColour'].fillna('') + " " +
        styles_df['productDisplayName'].fillna('')
)

# Initialize TF-IDF Vectorizer and create matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(styles_df['combined_features'])


# --- Recommendation Function ---
def get_recommendations(query, n=5):
    query_vec = tfidf.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-n:][::-1]
    return styles_df.iloc[top_indices]


# --- Streamlit UI ---
st.title("🛍️ Product Recommendation System")
st.markdown("Search for clothing products by typing or speaking!")

# Initialize session state for voice query
if 'voice_query' not in st.session_state:
    st.session_state.voice_query = ""

# Search bar
search_query = st.text_input("Enter product name or description:",
                             st.session_state.voice_query)

# Voice search button
with st.expander("🎤 Voice Search"):
    recognizer = sr.Recognizer()
    if st.button("Start Recording"):
        with st.spinner("Listening... Speak now!"):
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source, timeout=5)

                text_query = recognizer.recognize_google(audio)
                st.session_state.voice_query = text_query
                st.rerun()  # Refresh to show the recognized text

            except sr.UnknownValueError:
                st.error("Could not understand audio. Please try again.")
            except sr.RequestError as e:
                st.error(f"Could not reach Google Speech service: {e}")
            except Exception as e:
                st.error(f"Error during voice recognition: {e}")

# Display results
if search_query:
    results = get_recommendations(search_query)

    st.subheader(f"Recommended products for: '{search_query}'")
    for _, row in results.iterrows():
        col1, col2 = st.columns([1, 2])

        with col1:
            img_path = f"images/{row['id']}.jpg"
            if os.path.exists(img_path):
                st.image(img_path, width=150)
            else:
                st.warning("Image not found")

        with col2:
            st.write(f"**{row['productDisplayName']}**")
            st.write(f"- **Category:** {row['masterCategory']} > {row['subCategory']}")
            st.write(f"- **Type:** {row['articleType']}")
            st.write(f"- **Color:** {row['baseColour']}")
            st.write(f"- **Gender:** {row['gender']}")

        st.divider()

elif st.session_state.voice_query:
    # This ensures rerun after voice query but before searching
    search_query = st.session_state.voice_query
