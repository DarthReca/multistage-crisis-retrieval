import numpy as np
import pandas as pd
import random
import os
import torch
import json

from transformers import set_seed
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from langdetect import detect, DetectorFactory


def set_manual_seed(seed: int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed)
    DetectorFactory.seed = seed

def load_query_data(json_path: str="./json_files/query_mappings.json") -> json:
    """ Loads query mappings from a JSON file."""
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Query mappings file not found: {json_path}")
    
    # Load JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_results(dataframe: pd.DataFrame, csv_path: str, json_path: str=None):
    """Saves data as CSV and optionally as JSON."""
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    dataframe.to_csv(csv_path, index=False)

    if json_path:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        selected_json = [
            {
                "fact_number": i + 1,
                "query": row['query'],
                "text": row['text'],
                "search-text": row['search_text'],
                "cross-score": row['cross-score'],
                "retrieval-score": row['dense-score'],
                "label": row['label'],
                "type": row['source_type'],
            }
            for i, row in dataframe.iterrows()
        ]
        with open(json_path, "w") as f:
            json.dump(selected_json, f, indent=4)

def log_statistics(event_no, day, dataAsDF, filtered_data, selected_data_bm25, selected_data, relevant_data_topic_1, log):
        if log:
            print(f"Event {event_no} - Day {day}")
            print(f"\tlen data {len(dataAsDF)} -> len filtered {len(filtered_data)} -> len topic 1 {len(relevant_data_topic_1)} -> len bm25 {len(selected_data_bm25)} -> len dense {len(selected_data)} ")


def create_topic_model(
        embedding_model_name: str="crisistransformers/CT-mBERT-SE",
        device: str="cuda",
        n_neighbors: int=40,
        n_components: int=20,
        min_dist: float=0.1,
        min_cluster_size: int=40,
        min_samples: int=20,
        nr_topics: str="auto",
        allow_single_cluster: bool=False,
        top_n_words: int=12,
        diversity: float=0.5,
        max_features:int=50000,
        part_of_speech: str="en_core_web_md",
        verbose: bool=False
    ) -> BERTopic:
    """Instantiate a BERTopic model with configurable parameters."""

    # Define Representation Models
    main_representation_model = KeyBERTInspired(top_n_words=top_n_words)
    aspect_representation_model1 = PartOfSpeech(part_of_speech)
    aspect_representation_model2 = [
        KeyBERTInspired(top_n_words=10),
        MaximalMarginalRelevance(diversity=diversity)
    ]

    representation_model = {
        "Main": main_representation_model,
        "Aspect1": aspect_representation_model1,
        "Aspect2": aspect_representation_model2
    }

    # UMAP Configuration
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric="cosine",
        random_state=42
    )

    # HDBSCAN Clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        allow_single_cluster=allow_single_cluster
    )

    # Vectorizer for Text Representation
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_features=max_features
    )

    # Sentence Transformer Embeddings
    embedding_model = SentenceTransformer(embedding_model_name, device=device)

    # Instantiate BERTopic
    topic_model = BERTopic(
        nr_topics=nr_topics,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        verbose=verbose,
        language="english",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        top_n_words=top_n_words
    )

    return topic_model

def boost_scores(type_of_fact:str, score: float) -> float:
    """Boosts scores based on the type of fact."""
    
    if type_of_fact == 'News':
        return min(score * 1.5, 1.0)
    elif type_of_fact == 'Facebook':
        return min(score * 1.8, 1.0)
    elif type_of_fact == 'Twitter':
        return min(score * 1.8, 1.0)
    elif type_of_fact == 'Reddit':
        return min(score * 1.0, 1.0)
    else:
        return score

def get_language(text: str) -> str:
    """Detects the language of the given text."""
    
    if not isinstance(text, str) or pd.isna(text):
        return 'unknown'
    try:
        return detect(text)
    except:
        return 'unknown'