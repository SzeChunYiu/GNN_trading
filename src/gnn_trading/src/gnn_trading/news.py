import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, Optional
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

def aggregate_news_embeddings(news_csv: str, date_col='date', text_col='headline', model_name='ProsusAI/finbert', max_per_day=20) -> Dict[str, np.ndarray]:
    df = pd.read_csv(news_csv, parse_dates=[date_col]).sort_values(date_col)
    if text_col not in df.columns:
        raise ValueError(f"text_col='{text_col}' not in news csv")
    if not _HAS_TRANSFORMERS:
        # fallback: simple length & punctuation features
        out = defaultdict(list)
        for _, r in df.iterrows():
            day = str(pd.to_datetime(r[date_col]).date())
            txt = str(r[text_col])
            vec = np.array([len(txt), txt.count('!'), txt.count('?')], dtype=float)
            out[day].append(vec)
        return {k: np.mean(v, axis=0) for k,v in out.items()}

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    clf = pipeline('sentiment-analysis', model=mdl, tokenizer=tok, truncation=True)

    # batch process by day
    grouped = df.groupby(df[date_col].dt.date)
    day_vecs = {}
    for day, sub in grouped:
        texts = sub[text_col].astype(str).tolist()[:max_per_day]
        scores = clf(texts)
        # FinBERT labels: {'positive','negative','neutral'}
        vecs = []
        for s in scores:
            label = s['label'].lower()
            score = s['score']
            if 'pos' in label: vecs.append([score,0,0])
            elif 'neg' in label: vecs.append([0,score,0])
            else: vecs.append([0,0,score])
        day_vecs[str(day)] = np.mean(np.array(vecs), axis=0)
    return day_vecs
