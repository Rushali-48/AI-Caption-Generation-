from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import requests
import re

from feature_engineering import extract_features,E5_model

# load model
cat_model = joblib.load("ER_potential.pkl")

# load feature col
features = joblib.load("feature_columns.pkl")

# Init FASTAPI
app = FastAPI(title= "AI caption generation")

# request schema
class CaptionRequest(BaseModel):
    description: str
    followers: int
    total_posts: int 
    duration_sec: int
    datetime: str 

# generate caption using Mistral llama.cpp server
def generate_captions(prompt):
    full_prompt = f"""
Generate EXACTLY 3 viral Instagram unique captions and hashtags.

1.
caption text
#hashtags

2.
caption text
#hashtags

3.
caption text
#hashtags

Rules:
- Under 20 words
- use 4 to 6 hashtags
- Use emojis naturally
- English or Hinglish
- Make captions catchy and trendy
- Avoid generic captions
- Avoid repeating words
- Avoid repeating hashtags
- Use different hashtag styles
- Use modern Instagram slang naturally
- Output ONLY:
caption
hashtags

If you do not follow the format exactly, the answer is incorrect.
 
Topic:
{prompt}
"""
    payload = {
         "messages": [
        {
            "role": "system",
            "content": (
                "You strictly follow formatting instructions "
                "and generate viral Instagram captions."
            )
        },
        {
            "role": "user",
            "content": full_prompt
        }
    ],
        "temperature": 0.80,
        "max_tokens": 250,
        "top_p": 0.92,
        "repeat_penalty": 1.2,
        "stream": False
    }

    headers = {
        "Content-Type": "application/json",
        "Connection": "close"
    }

    print("Calling llama.cpp server...")

    response = requests.post(
        "http://127.0.0.1:8001/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=300
    )

    print("Status code:", response.status_code)
    print(response.text[:300])

    data = response.json()

    print("LLM response received")

    return data["choices"][0]["message"]["content"]

# parse LLM output
def parse_output(text):

    results = []

    blocks = re.split(r"\n\s*\d+\.\s*", text)

    for block in blocks:

        block = block.strip()

        if not block:
            continue

        hashtags = re.findall(r"#\w+", block)

        caption = re.sub(r"#\w+", "", block).strip()

        caption = re.sub(r"^caption\s*:\s*", "", caption, flags=re.IGNORECASE)
        caption = re.sub(r"^\d+\.\s*", "", caption)


        if caption and hashtags:

            if len(caption.split()) < 2:
                continue

            results.append({
                "caption": caption,
                "hashtags": " ".join(hashtags)
            })

    return results[:3]
    
# Main API Endpoint
@app.post("/generate")
async def generate(data: CaptionRequest):

    try:

        print("API endpoint called")

        raw_output = generate_captions(data.description)

        print("RAW OUTPUT:")
        print(raw_output)

        caption_data = parse_output(raw_output)

        print("PARSED DATA:")
        print(caption_data)

        if len(caption_data) == 0:
            return {
                "error": "LLM output parsing failed",
                "raw_output": raw_output
            }

        captions = [item["caption"] for item in caption_data]
        hashtags = [item["hashtags"] for item in caption_data]

        caption_inputs = ["passage: " + text for text in captions]
        hashtag_inputs = ["passage: " + text for text in hashtags]

        print("Generating embeddings...")

        cap_embs = E5_model.encode(
            caption_inputs,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        hash_embs = E5_model.encode(
            hashtag_inputs,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        print("Embeddings generated")

        ranked_result = []

        for idx, item in enumerate(caption_data):

            print(f"Processing caption {idx}")

            feature_df = extract_features(
                caption=item["caption"],
                hashtags=item["hashtags"],
                cap_emb=cap_embs[idx],
                hash_emb=hash_embs[idx],
                followers_count=data.followers,
                total_posts=data.total_posts,
                duration_sec=data.duration_sec,
                datetime_str=data.datetime
            )

            feature_df = feature_df.reindex(columns=features,fill_value=0)

            print(feature_df.shape)

            score = cat_model.predict(feature_df)[0]

            score = max(0, min(score, 1))

            ranked_result.append({
                "caption": item["caption"],
                "hashtags": item["hashtags"],
                "engagement_score": round(float(score) * 100, 2)
            })

        ranked_results = sorted(
            ranked_result,
            key=lambda x: x["engagement_score"],
            reverse=True
        )

        return {
            "best_caption": ranked_results[0],
            "all_captions": ranked_results
        }

    except Exception as e:

        print("ERROR:")
        print(str(e))

        return {"error": str(e)}
