# Emotion-Based Film Recommendation System

This project is a unique film recommendation engine that learns from personal emotional experiences and viewer-written film reviews.

Unlike traditional systems, it does not rely on metadata such as cast, release dates, or genre. Instead, it recommends films based solely on the emotional experience a user seeks.

---

## Overview

This system asks the user:

"What do you want to feel while watching a film?"

Based on this emotional intent, it analyzes viewer-written reviews and recommends films that are emotionally aligned with the input.

---

## How It Works

1. The developer provides personal experiences that describe each emotion (e.g., nostalgia, confusion, melancholy).
2. These experiences are converted into vector embeddings using a SentenceTransformer model.
3. Film reviews are also embedded in the same vector space.
4. A FAISS-powered similarity search matches user intent with emotionally similar film reviews.
5. The most aligned reviews are selected, and the corresponding films are recommended.

---

## Technologies Used

- **SentenceTransformer** for semantic embedding
- **FAISS** for fast nearest neighbor search
- **Gradio** for the user interface
- **pandas** and **numpy** for data processing

---

## Directory Structure

project/
├── main.py # Main logic for RAG system
├── hisdeneyim.py # GUI to define emotion experiences
├── film_yorumlari.csv # Viewer-written reviews
├── his_deneyimleri.csv # Developer-defined emotion experiences
├── requirements.txt # Python dependencies


---

## Installation

### Step 1: Create the environment

```bash
conda create -n ragfilm python=3.10
conda activate ragfilm
Step 2: Install the required packages
conda install -c pytorch faiss-cpu
pip install -r requirements.txt
Running the System
To start the emotion-driven film recommender:
python main.py
A local interface will open at http://127.0.0.1:7860 using Gradio.

Philodophy and Ethics
Emotions are not predefined or abstractly categorized.

The AI learns what a given emotion means only through real, lived experiences provided by the developer.

Films are not judged by objective metrics, but by how they made viewers feel.

This allows the system to remain subjective, contextual, and closer to human emotional reasoning.

Future Improvements
Add active learning and feedback from users

Fine-tune embeddings using user input

Expand to multi-label emotional tagging

Implement multilingual support

License
This project is for educational and non-commercial use only. Please respect user privacy and data ethics when expanding or deploying this system.


---




## 📧 Contact

Project Link: [https://github.com/tunahadimioglu/messi-ai](https://github.com/tunahadimioglu/messi-ai)

---

⭐ If you found this project helpful, please give it a star!
