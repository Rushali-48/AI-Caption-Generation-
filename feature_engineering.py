import re
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
import emoji
from textblob import TextBlob

# load model 
E5_model = SentenceTransformer("E5_model")

def extract_features(caption,hashtags,cap_emb,hash_emb,followers_count,total_posts,duration_sec,datetime_str):

    # Extract hashtag
    hashtag = re.findall(r"#\w+", hashtags)
    cleaned = []
    for tag in hashtag:
        tag = tag.lower().strip()
        tag = tag.replace("#","")

        if tag:
            cleaned.append(tag)
    
    cleaned = list(dict.fromkeys(cleaned))
    hashtags = " ".join(cleaned)
    hashtag_count = len(cleaned)

    # Extract emoji
    emoji_list = [c for c in caption if c in emoji.EMOJI_DATA]
    emoji_count = len(emoji_list)

    # clean caption text
    text = re.sub(r"[\u200b-\u200d\uFEFF]", "",caption) # remove invisible unicode chars
    text = re.sub(r'([!?.,])\1{3,}', r'\1\1', text) # remove excessive puncutuation
    text = re.sub(r"\s+", " ", text).strip()

    full_text_clean = text
    caption_length = len(full_text_clean.split()) if full_text_clean else 1
    sentiment = TextBlob(full_text_clean).sentiment.polarity

    # Create Density 
    emoji_density = emoji_count/(caption_length + 1)
    hashtag_density = hashtag_count/(caption_length + 1)

    # follower normalization
    followers_log = np.log1p(followers_count)
    
    # Post Normalization
    post_log = np.log1p(total_posts)

    # Hook,CTA feature

    curiosity_hooks = ["wait", "watch till end","don't skip", "you won't believe","what happened next","unexpected","plot twist","ending","secret","finally"]
    relatable_hooks = [ "pov","bro","every girl","every boy","middle class","real struggle","only legends","relatable","parents","best friend"]
    emotional_hooks = ["cry","heartbroken","miss you","sad","emotional","pain","love","broken"]
    hype_hooks = ["viral","crazy","insane","omg","wtf","unbelievable","legendary","fire"]
    cta_words = ["comment","share","follow","save","tag","send this"]

    def count_keyword(text,keyword):
        text = text.lower()

        return sum(kw in text for kw in keyword)
    
    curiosity_score = count_keyword(full_text_clean,curiosity_hooks)
    relatable_score = count_keyword(full_text_clean,relatable_hooks)
    emotional_score = count_keyword(full_text_clean,emotional_hooks)
    hype_score = count_keyword(full_text_clean,hype_hooks)
    cta_score = count_keyword(full_text_clean,cta_words)

    # Extract Time Features
    dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    hour = dt.hour
    weekday = dt.weekday()
    is_weekend = int(weekday in [5,6])

    # similarity
    caption_hashtag_similarity = float(np.dot(cap_emb, hash_emb))

    # Base feature
    feature_dict = {
        'hashtag_count': hashtag_count,
        'emoji_count': emoji_count,
        'duration_sec': duration_sec,
        'followers_log': followers_log,
        'caption_length': caption_length,
        'emoji_density': emoji_density,
        'hashtag_density': hashtag_density,
        'sentiment': sentiment,
        'curiosity_score': curiosity_score,
        'relatable_score': relatable_score,
        'emotional_score': emotional_score,
        'hype_score': hype_score,
        'cta_score': cta_score,
        'post_log': post_log,
        'hour': hour,
        'weekday': weekday,
        'is_weekend': is_weekend,
        'caption_hashtag_similarity': float(caption_hashtag_similarity)
    }

    # add embeddings
    for i,val in enumerate(cap_emb):
        feature_dict[f"cap_emb_{i}"] = float(val)
        
    for i, val in enumerate(hash_emb):
        feature_dict[f"hash_emb_{i}"] = float(val)

    return pd.DataFrame([feature_dict])
