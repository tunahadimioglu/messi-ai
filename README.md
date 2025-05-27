# 🎬 Emotional Movie Recommendation AI

This project is an artificial intelligence application that learns users' emotional experiences and provides personalized movie recommendations based on this information.

## 🌟 Features

- **Emotion Learning**: AI learns your defined emotional experiences
- **Movie Analysis**: Analyzes and categorizes emotions in movie reviews
- **Personalized Recommendations**: Suggests movies based on the emotion you want to feel
- **AI Test System**: Test how well the AI has learned
- **Data Persistence**: Learned data is automatically saved

## 🎭 Supported Emotions

- **Happiness**: Joy, cheerfulness, excitement
- **Sadness**: Sorrow, melancholy, pain
- **Fear**: Anxiety, terror, tension
- **Anger**: Rage, fury, wrath
- **Love**: Romance, affection, passionate attachment
- **Excitement**: Adventure, action, thrill
- **Nostalgia**: Longing for the past, remembrance
- **Hope**: Optimism, future expectations
- **Lust**: Passion, desire, sexual attraction
- **Regret**: Remorse, guilt, conscience
- **Inadequacy**: Feeling insufficient, insecurity
- **Shame**: Embarrassment, bashfulness
- **Anxiety**: Worry, stress, nervousness
- **Hatred**: Hostility, disgust
- **Jealousy**: Envy, resentment
- **Attachment**: Love, closeness, trust

## 🚀 Installation

### Requirements
- Python 3.8+
- pip (Python package manager)

### Step 1: Download the Project
```bash
git clone https://github.com/tunahadimioglu/messi-ai.git
cd messi-ai
```

### Step 2: Create Virtual Environment
```bash
# For Windows
python -m venv myenv
myenv\Scripts\activate

# For macOS/Linux
python3 -m venv myenv
source myenv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### First Run
```bash
python3 main.py
```

### AI Training Process
1. **Emotion Learning**: Describe your personal experiences for each emotion
2. **Movie Data Input**: Enter movie names and reviews
3. **Testing**: Test how well the AI has learned

### Main Menu Options
- **1**: Get movie recommendation
- **2**: Add new movie data
- **3**: Test AI
- **4**: Exit

## 📁 File Structure

```
emotional-movie-ai/
├── main.py                    # Main application file
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── emotion_ai_model.pkl      # Trained AI model (auto-generated)
├── user_emotion_data.json    # User emotion data (auto-generated)
├── movie_data.json          # Movie data (auto-generated)
└── myenv/                   # Virtual environment folder
```

## 🧠 How Does the AI Work?

1. **Learning**: Learns word patterns from your emotional experiences
2. **Analysis**: Detects emotions in texts using TF-IDF and Machine Learning
3. **Matching**: Categorizes movie reviews into emotional categories
4. **Recommendation**: Suggests the most suitable movie for your desired emotion

## 🔧 Development

### Adding New Emotions
You can add new emotions to the `emotion_tags` dictionary in `main.py`:

```python
emotion_tags = {
    "new_emotion": {
        "description": "Description",
        "examples": ["example1", "example2"]
    }
}
```

### Model Reset
If you want to reset the AI:
```bash
rm emotion_ai_model.pkl user_emotion_data.json movie_data.json
```


## 📧 Contact

Project Link: [https://github.com/tunahadimioglu/messi-ai](https://github.com/tunahadimioglu/messi-ai)

---

⭐ If you found this project helpful, please give it a star!
