# AI Viral Caption Generation

An end-to-end Generative AI + Machine Learning system that automatically generates viral social media captions and predicts their engagement potential using NLP embeddings, engagement modeling, and LLM-powered caption generation.

---

## Project Overview

This project combines:

- 📊 YouTube data collection pipeline
- 🧠 NLP feature engineering
- 🤖 LLM-based caption generation
- 📈 Engagement potential prediction
- ⚙️ FastAPI deployment


The system is designed to:

1. Collect high-performing social media content

2. Analyze caption patterns and engagement behavior

3. Generate viral-style captions using an LLM

4. Rank generated captions using a trained ML model

5. Return the best-performing caption prediction

---

## Core Technologies

| Component           | Technology                                     |
| ------------------- | ---------------------------------------------  |
| Backend API         | FastAPI                                        |
| ML Model            | CatBoost Regressor                             |
| NLP Embeddings      | Multilingual E5 Transformer                    |
| LLM Inference       | qwen2.5-1.5b-instruct-q4_k_m.gguf via llama.cpp|
| Feature Engineering | pandas + NumPy                                 |
| Dataset Source      | YouTube Data API                               |
| Deployment          | Python                                         |

---

## Project Structure


## Key Features

### 1. Data Collection Features

- Seed Query Expansion 
- Video Filtering
- Extracted Metadata 

---

### 2. Feature Engineering Pipeline

- Text Features 
- Density Features
- Time-Based Features 

---

### 3. Semantic Features

- Multilingual E5 Embeddings
- Similarity to calculates aption ↔ hashtag semantic similarity

---

## Model Details

- Model: CatBoost Regressor 

### Why CatBoost Regressor ?

- Handles tabular + embedding features well
- Strong regression performance
- Robust with noisy engagement data
- Better ranking behavior  

---

## Target Variable

- The model predicts: engagement_score (ER.rank(pct=True))
- derived from:  ER = (likes + comments) / views

---

## Evaluation Metrics

The model is evaluated using:

- MAE
- RMSE
- R² Score
- Spearman Correlation

---

## LLM-Based Caption Generation

- The system uses: qwen2.5-1.5b-instruct-q4_k_m.gguf via llama.cpp to generate viral Instagram captions.

### Generation Rules

The LLM is instructed to:

- Generate exactly 3 captions
- Use emojis naturally
- Generate trendy hashtags
- Avoid repetitive hashtags
- Keep captions under 20 words
- Produce catchy and modern captions

---

## Caption Ranking Pipeline
Flow: 

---

# FastAPI Backend
- API Request

{
  "description": "Gym Tranformation vedio",
  "followers": 1200,
  "total_posts": 30,
  "duration_sec": 60,
  "datetime": "2026-05-16 20:00:00"
}

---

# API Response
- recommended caption


---

## How to Run

1. create working environment

2. pip install -r requirements.txt  

3. .\build\bin\llama-server.exe -m "D:\Rushali Projects\Zenith internship project\caption_generation\models\qwen2.5-1.5b-instruct-q4_k_m.gguf" -c 1048 --threads 4 --port 8001 (to run in you system first install LLM model and llama.ccp dependencies seperatly and after using original llm model path run llm in local server)

4. uvicorn main:app --reload  

5. streamlit run app.py  

---

## Requirements

- fastapi
- uvicorn
- streamlit
- requests
- pandas
- numpy
- sentence-transformers
- scikit-learn
- catboost
- emoji
- textblob
- joblib
- scipy

---

## Future Improvements

- Streamlit dashboard
- Multi-platform caption generation
- Real-time trend detection
- Reinforcement learning for ranking
- Multi-language caption generation
- Fine-tuned local LLM

---

## Key Insight

- This project predicts: “How strong is this generated caption likely to perform?”
- Not: “Will this definitely go viral?”