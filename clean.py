import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z\d\s.,!?]', '', text, flags=re.UNICODE)  # keep letters, digits, punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # normalize spaces
    return text