from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import openai

# Settings
openai.api_key = "YOUR_OPENAI_API_KEY"
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load data
emotion_df = pd.read_csv("his_deneyimleri.csv")  # Columns: Emotion, Experience
review_df = pd.read_csv("film_yorumlari.csv")    # Columns: Film, Review

# 1. Get all emotion experience embeddings
emotion_embeddings = model.encode(emotion_df['Experience'].tolist(), show_progress_bar=True)

# 2. Get all film review embeddings
review_embeddings = model.encode(review_df['Review'].tolist(), show_progress_bar=True)

# 3. Build FAISS index for reviews
dim = review_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(review_embeddings))

# 4. Retrieve top-k closest reviews for a given emotional input

def find_closest_reviews(user_emotions, top_k=5):
    user_vector = model.encode(user_emotions, show_progress_bar=False)
    if isinstance(user_vector[0], list) or isinstance(user_vector[0], np.ndarray):
        avg_vector = np.mean(user_vector, axis=0)
    else:
        avg_vector = user_vector
    distances, indices = index.search(np.array([avg_vector]), top_k)
    return review_df.iloc[indices[0]]

# 5. Ask GPT for film recommendations

def get_recommendation_from_gpt(emotion_label, user_experiences, matched_reviews_df):
    review_text = "\n\n".join([
        f"{row['Film']}: {row['Review']}" for _, row in matched_reviews_df.iterrows()
    ])
    prompt = f"""
The user wants to experience the following emotion(s): {emotion_label}

These emotions were described through the following personal experiences:
---
{chr(10).join(user_experiences)}
---

Below are viewer reviews that closely match these experiences:
===
{review_text}
===

Based on the above, recommend 2 films that are likely to evoke these emotions.
Include a short explanation for each recommendation.
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response['choices'][0]['message']['content']

# 6. Example simulation: user inputs emotional goals
user_inputs = [
    "I want to be emotionally overwhelmed after the movie",
    "I want to feel melancholic"
]

# Find closest personal experiences for each emotional goal
matched_experiences = []
matched_labels = []

for input_text in user_inputs:
    input_vector = model.encode(input_text)
    sim_vectors = model.encode(emotion_df['Experience'].tolist())
    similarities = np.dot(sim_vectors, input_vector) / (
        np.linalg.norm(sim_vectors, axis=1) * np.linalg.norm(input_vector)
    )
    best_idx = np.argmax(similarities)
    matched_experiences.append(emotion_df.iloc[best_idx]['Experience'])
    matched_labels.append(emotion_df.iloc[best_idx]['Emotion'])

# Retrieve matching reviews and get GPT recommendation
matched_reviews = find_closest_reviews(matched_experiences, top_k=6)
recommendation = get_recommendation_from_gpt(", ".join(matched_labels), matched_experiences, matched_reviews)

print("\n\nRecommended Films:\n")
print(recommendation)
