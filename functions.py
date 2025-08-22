from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools
import pandas as pd
import re
import torch
from tqdm import tqdm
import numpy as np
from transformers import pipeline
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from keybert import KeyBERT

stopwords = set(STOPWORDS)
words = {"video", "watch", "channel", "subscribe", "like", "please", "nvidia", "gpu", "card", "will", "game", "lol", "im"}
stopwords.update(words)

def generate_wordcloud(df, sentiment):
    text = " ".join(df[df['transformer_label'] == sentiment]['comment'].dropna().astype(str))
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=stopwords,
        collocations=False
    ).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"WordCloud for {sentiment} Comments", fontsize=16)
    plt.show()

def sarcasm_to_binary(x):
    if not isinstance(x, str):
        return 0  # fallback if somehow NaN

    text = x.strip().lower()

    # Treat "normal" as not sarcastic
    if text == "normal":
        return 0

    # If the string looks like junk repeating tokens (derison, risonel, etc.)
    if re.search(r"(derison|risonel|rison)", text):
        return 1

    # Any non-normal string (like "shut up", "sarcasm", "yeah right", etc.) → sarcastic
    return 1

def classify_emotions(texts, model_name="SamLowe/roberta-base-go_emotions", batch_size=64, top_k=5):
    """
    Classify emotions for a list of texts using HuggingFace GoEmotions model.

    Args:
        texts (list[str]): List of text comments.
        model_name (str): HuggingFace model name.
        batch_size (int): Number of texts to process per batch.
        top_k (int): Number of top emotions to return per text.

    Returns:
        list[list[str]]: A list where each element is a list of top_k emotions for the text.
    """
    # Load model only once
    classifier = pipeline("text-classification",
                          max_length=180,
                          model=model_name, 
                          top_k=None,
                          truncation=True,
                          return_all_scores=True, 
                          device=0 if torch.cuda.is_available() else -1
                         )  # set device=-1 for CPU

    all_emotions = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        res = classifier(batch)

        # Process each text in the batch
        for r in res:
            sorted_res = sorted(r, key=lambda x: x["score"], reverse=True)[:top_k]
            all_emotions.append([x["label"] for x in sorted_res])

    return all_emotions

def visualize_emotions(df, top_emotion=None, text_col="comment"):
    """
    Visualize emotions in YouTube comments.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'emotions' column (list of labels per comment).
    top_emotion : str
        Used when kind="wordcloud" -> generate wordcloud for that emotion.
    text_col : str
        Column containing the text (default="comment").
    """
    if not top_emotion:
            raise ValueError("You must specify `top_emotion` for wordcloud visualization.")

    text = " ".join(df[df["emotions"].apply(lambda x: top_emotion in x)][text_col])

    wordcloud = WordCloud(width=800, height=400,
                              background_color="white",
                              stopwords=stopwords).generate(text)
    plt.figure(figsize=(10,6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud for {top_emotion}", fontsize=16)
    plt.show()

def topic_modeling(df, text_col="comment", n_topics=None):
    """
    Perform topic modeling on YouTube comments using BERTopic.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing comments.
    text_col : str
        Column containing the text.
    n_topics : int or None
        Number of topics (None = let BERTopic decide automatically).
    """

    # Mask for non-empty comments
    mask = df[text_col].notna()
    comments = df.loc[mask, text_col].tolist()

    # Initialize BERTopic
    vectorizer_model = CountVectorizer(stop_words="english")
    topic_model = BERTopic(vectorizer_model=vectorizer_model, nr_topics=n_topics)

    # Fit and transform
    topics, probs = topic_model.fit_transform(comments)

    # Create topic column with NaNs, then assign back
    df["topic"] = np.nan
    df.loc[mask, "topic"] = topics

    # Get topic info
    topic_info = topic_model.get_topic_info()

    return df, topic_model, topic_info

# Initialize KeyBERT with a transformer model
kw_model = KeyBERT(model="all-MiniLM-L6-v2")

def extract_keywords_per_sentiment(df, text_col="comment", sentiment_col="sentiment", 
                                    top_n=10, ngram_range=(1,2)):
    """
    Extracts keywords per sentiment class using KeyBERT.
    
    Args:
        df (pd.DataFrame): DataFrame containing comments and sentiment labels
        text_col (str): Name of column with text
        sentiment_col (str): Name of sentiment label column
        top_n (int): Number of keywords to extract
        ngram_range (tuple): N-gram range for keyword extraction
    
    Returns:
        dict: {sentiment: [(keyword, score), ...]}
    """
    sentiment_keywords = {}
    
    for sentiment in df[sentiment_col].unique():
        subset = df[df[sentiment_col] == sentiment][text_col].dropna().tolist()
        
        # Combine all comments for that sentiment
        joined_text = " ".join(subset)
        
        keywords = kw_model.extract_keywords(
            joined_text,
            keyphrase_ngram_range=ngram_range,
            stop_words="english",
            top_n=top_n
        )
        sentiment_keywords[sentiment] = keywords
    
    return sentiment_keywords