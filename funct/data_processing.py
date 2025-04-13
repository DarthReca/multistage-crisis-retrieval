import os
import torch
import nltk
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from sentence_transformers import util
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
from collections import Counter
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

from funct.utils import boost_scores, get_language

def build_bm25_index(dataAsDF):
    """Build BM25 index on the dataset."""
    try:
        nltk.word_tokenize("test")
    except LookupError:
        nltk.download("punkt")
    
    tokenized_corpus = [nltk.word_tokenize(text.lower()) for text in dataAsDF["search_text"].tolist()]
    return BM25Okapi(tokenized_corpus), tokenized_corpus

def get_embeddings(text_list, model, batch_size, device):
    """Generate embeddings for a list of texts."""
    return model.encode(text_list, 
                        convert_to_tensor=True, 
                        show_progress_bar=False, 
                        device=device, 
                        batch_size=batch_size, 
                        normalize_embeddings=True, 
                        clean_up_tokenization_spaces=True)
    
def extract_keywords(text, top_n=5):
    kw_model = KeyBERT()
    """Extract keywords from a query."""
    keywords = kw_model.extract_keywords(text, top_n=top_n, keyphrase_ngram_range=(1, 3), stop_words='english')
    return [kw[0] for kw in keywords]

def compute_relevant_topics(topics, topic_model, event_embeddings, sentence_retriever, filtered_data, add_outliers=False):
    relevant_topics = []
    for topic_idx in set(topics):
        topic_words = topic_model.get_topic(topic_idx)
        if isinstance(topic_words, bool):
            continue
        topic_words = [word for word, _ in topic_words]
        topic_embedding = sentence_retriever.encode(" ".join(topic_words), convert_to_tensor=True, normalize_embeddings=True).cpu().numpy()
        similarity_score = cosine_similarity([event_embeddings], [topic_embedding])[0][0]
        if not add_outliers and topic_idx == -1:
            continue
        relevant_topics.append({"topic": topic_idx, "score": similarity_score})
    
    return relevant_topics

def print_topic_details(topics, top_relevant_topics, relevant_topics, topic_model, topic_useful_dict, top_percentile):
    topic_counts = Counter(topics)

    print(f"\nTopic Details: top_percentile={top_percentile:.4f}")
    for topic_idx in set(topics):
        doc_count = topic_counts[topic_idx]
        useful_count = topic_useful_dict.get(topic_idx, 0)
        is_selected = any(topic['topic'] == topic_idx for topic in top_relevant_topics)
        similarity_entry = next((topic for topic in relevant_topics if topic['topic'] == topic_idx), None)
        sim_score = similarity_entry['score'] if similarity_entry else 0.0
        topic_words = topic_model.get_topic(topic_idx)
        if isinstance(topic_words, bool):
            continue
        topic_words = [word for word, _ in topic_words]

        if useful_count > 0 or is_selected:
            print(f"\tTopic {topic_idx} {'[YES]' if is_selected else '[_NOT]'}\tscore: ({sim_score:.4f})\t({doc_count} documents, {useful_count} USEFUL): \t{topic_words}")

def filter_low_tfidf_df(dataframe, threhsold=0.1):
    """ Remove documents with low TF-IDF variance (weak signal) """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(dataframe['search_text'].tolist())
    doc_scores = tfidf_matrix.sum(axis=1).A1  # Sum TF-IDF values per doc
    
    # Keep only documents with a strong signal
    valid_indices = doc_scores > threhsold
    return dataframe[valid_indices].reset_index(drop=True)

@torch.no_grad()
def assign_topics(filtered_data, topic_model):
    """Assign BERTopic topics to each document and store topic words in the DataFrame."""
    
    # filtered_docs = filter_low_tfidf_documents(filtered_data['search_text'].tolist())
    filtered_docs = filtered_data['search_text'].tolist()
    if not filtered_docs:
        print("No valid documents to process.")
        return filtered_data

    try:
        topics, _ = topic_model.fit_transform(filtered_docs)
        valid_indices = filtered_data['search_text'].isin(filtered_docs)
        filtered_data.loc[valid_indices, 'topic'] = topics
        
    except Exception as e:
        print(f"Error processing BERTopic: {e}")
        return filtered_data
    
    return filtered_data

def select_relevant_topics(filtered_data, queryAsDF, sentence_retriever, topic_model, index=0, top_percentile=75, threshold=0.5, print_details=False, top_words=20, add_outliers=False):
    """Select relevant topics based on affinity scores with event/query descriptions."""

    if filtered_data.empty or 'topic' not in filtered_data.columns:
        print("No topics assigned, skipping selection.")
        return filtered_data

    # Compute topic-document usefulness
    topic_useful_counts = (
        filtered_data[filtered_data['label'] == 'USEFUL']
        .groupby('topic')
        .size()
        .reset_index(name='useful_count')
    )
    topic_useful_dict = dict(zip(topic_useful_counts['topic'], topic_useful_counts['useful_count']))

    event_description = queryAsDF.loc[index, 'event_description']
    topic_event_description = np.squeeze(extract_keywords(event_description, top_n=top_words))
    # print(f"Event Description: {event_description}")
    # print(f"Event Keywords: {topic_event_description}")
    event_embeddings = sentence_retriever.encode(" ".join(topic_event_description), convert_to_tensor=True).cpu().numpy()

    # Compute topic similarity scores
    relevant_topics = compute_relevant_topics(filtered_data['topic'].tolist(), 
                                              topic_model, 
                                              event_embeddings, 
                                              sentence_retriever, 
                                              add_outliers=add_outliers, 
                                              filtered_data=filtered_data)

    # Select the most relevant topics
    top_percentile_value = np.percentile([topic['score'] for topic in relevant_topics], top_percentile)
    top_k_topics = [topic for topic in relevant_topics if topic['score'] > top_percentile_value and topic['score'] > threshold]
    
    # if top_k_topics is empty, select the top 1 topic based on the highest score
    if not top_k_topics:
        max_score = max(relevant_topics, key=lambda x: x['score'])['score']
        if max_score > threshold:
            top_k_topics = [max(relevant_topics, key=lambda x: x['score'])]
    
    if print_details:
        # print_topic_details(filtered_data['topic'].tolist(), relevant_topics, relevant_topics, topic_model, topic_useful_dict, top_percentile_value)
        print_topic_details(filtered_data['topic'].tolist(), top_k_topics, relevant_topics, topic_model, topic_useful_dict, top_percentile_value)

    return filtered_data[filtered_data['topic'].isin([topic['topic'] for topic in top_k_topics])].reset_index(drop=True)

def filter_redundant_data(dataAsDF, event_no=None, day=None):
    """Filter redundant documents using exact match and fuzzy deduplication (no embeddings required)."""

    # Load saved reduced dataset if exists
    if day is not None:
        file_path = f"./reduced_dataset/{event_no}_{day}_data.csv"
        if os.path.exists(file_path):
            print(f"Loading reduced data for {event_no} on {day}")
            return pd.read_csv(file_path)
        print(f"Reducing data for {event_no} on {day}")
        os.makedirs("./reduced_dataset", exist_ok=True)

    # Step 1: Pre-filter short texts and remove exact duplicates
    dataAsDF = dataAsDF.loc[
        (dataAsDF['search_text'].str.split().str.len() > 2) &  # Remove short texts
        (~dataAsDF['text'].str.contains(r"\?", regex=True))  # Remove questions
    ].drop_duplicates(subset=['search_text']).reset_index(drop=True)  # Exact duplicate removal

    # Step 2: Remove near-duplicates using string similarity
    unique_texts = set()
    filtered_rows = []
    
    for _, row in dataAsDF.iterrows():
        text = row['search_text'].strip().lower()
        if text not in unique_texts:
            unique_texts.add(text)
            filtered_rows.append(row)

    # Convert back to DataFrame
    filtered_data = pd.DataFrame(filtered_rows)
    
    # filter out the non-english text
    filtered_data['lang'] = filtered_data['search_text'].apply(get_language)
    filtered_data = filtered_data[filtered_data['lang'] == 'en']
    filtered_data.drop(columns=['lang'], inplace=True)
    filtered_data.dropna(inplace=True)
    
    filtered_data = filter_low_tfidf_df(filtered_data)

    # Save dataset
    if day is not None:
        filtered_data.to_csv(file_path, index=False)

    return filtered_data

def filter_semantic_similarity_2(df, embedding_model, text_column="search_text", threshold=0.9):
    """Filters out redundant text using cosine similarity of sentence embeddings."""
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        text_list = df[text_column].tolist()
        
        # Compute all embeddings in one batch (very fast)
        embeddings = embedding_model.encode(text_list, convert_to_tensor=True, normalize_embeddings=True)
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings.cpu().numpy())

        num_texts = len(text_list)
        keep_mask = np.ones(num_texts, dtype=bool)

        for i in tqdm(range(num_texts - 1), desc="Filtering Texts"):
            if keep_mask[i]:  
                # Find similar texts using cosine similarity
                similar_indices = np.where(similarity_matrix[i, i+1:] >= threshold)[0]
                keep_mask[i + 1 + similar_indices] = False  

    return df[keep_mask].reset_index(drop=True)

def process_queries_bm25(queryAsDF, dataAsDF, bm25, cross_encoder, top_k, alpha=0.0, dynamic_score=70):

    for _, query_row in tqdm(queryAsDF.iterrows(), total=len(queryAsDF), desc="Processing queries bm25"):
        
        # **Step 1: BM25 Retrieval**
        query_tokens = nltk.word_tokenize(query_row["revised_text"].lower())
        bm25_scores = bm25.get_scores(query_tokens)
        top_doc_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]

        # Extract BM25 scores
        dense_scores = [bm25_scores[idx] for idx in top_doc_indices]

        # **Step 2: Cross-Encoder Re-ranking**
        cross_inp = [[query_row["long_query"], dataAsDF.iloc[idx]["search_text"]] for idx in top_doc_indices]
        with torch.no_grad():
            cross_scores = cross_encoder.predict(cross_inp)
        
        # **Step 3: Combine Scores (BM25 + Cross-Encoder)**
        combined_scores = [
            alpha * dense_score + (1 - alpha) * cross_score
            for dense_score, cross_score in zip(dense_scores, cross_scores)
        ]

        # **Step 4: Sort by Combined Score**
        sorted_hits = sorted(
            zip(top_doc_indices, combined_scores, cross_scores, dense_scores),
            key=lambda x: x[1],  # Sort by combined score
            reverse=True
        )
        
        dynamic_min_score = np.percentile(combined_scores, dynamic_score)
        
        # **Step 5: Store Selected Data**
        for doc_idx, combined_score, cross_score, dense_score in sorted_hits:
            current_score = dataAsDF.iloc[doc_idx]["cross-score"]
            
            type_of_fact = dataAsDF.iloc[doc_idx]['source_type']
            combined_score = boost_scores(type_of_fact, combined_score)
            
            if combined_score > current_score:
                
                dataAsDF.at[doc_idx, "query"] = query_row["long_query"]
                dataAsDF.at[doc_idx, "query_id"] = query_row["query_id"]
                dataAsDF.at[doc_idx, "cross-score"] = combined_score
                dataAsDF.at[doc_idx, "dense-score"] = dense_score
                dataAsDF.at[doc_idx, "reranker-score"] = cross_score
                if combined_score > dynamic_min_score:
                    dataAsDF.at[doc_idx, "is_selected"] = True

def process_queries_dense(queryAsDF, data_embeddings, dataAsDF, sentence_retriver, cross_encoder, top_k, batch_size, device, dynamic_score, alpha=0.0):

    for _, query_row in tqdm(queryAsDF.iterrows(), total=len(queryAsDF), desc="Processing queries dense"):
        # Step 1: Dense Retrieval
        query_embedding = get_embeddings([query_row['revised_text']], sentence_retriver, batch_size, device)
        hits = util.semantic_search(query_embedding, data_embeddings, top_k=top_k)[0]
        
        # Extract dense scores and document indices
        dense_scores = [hit['score'] for hit in hits]
        doc_indices = [hit['corpus_id'] for hit in hits]

        # Step 2: Cross-Encoder Re-ranking
        cross_inp = [[query_row['long_query'], dataAsDF.iloc[idx]['search_text']] for idx in doc_indices]
        with torch.no_grad():
            cross_scores = cross_encoder.predict(cross_inp)
        
        # Step 3: Combine Scores
        combined_scores = [
            alpha * dense_score + (1 - alpha) * cross_score
            for dense_score, cross_score in zip(dense_scores, cross_scores)
        ]

        # Step 4: Sort by Combined Score
        sorted_hits = sorted(
            zip(doc_indices, combined_scores, cross_scores, dense_scores),
            key=lambda x: x[1],  # Sort by combined score
            reverse=True
        )

        # Step 5: Thresholding and Updating DataFrame
        dynamic_min_score = np.percentile(combined_scores, dynamic_score)  # Dynamic thresholding
        
        for doc_idx, combined_score, cross_score, dense_score in sorted_hits:
            current_score = dataAsDF.iloc[doc_idx]['cross-score']
            
            type_of_fact = dataAsDF.iloc[doc_idx]['source_type']
            combined_score = boost_scores(type_of_fact, combined_score)
            
            if combined_score > current_score:
                dataAsDF.at[doc_idx, 'query'] = query_row['long_query']
                dataAsDF.at[doc_idx, 'query_id'] = query_row['query_id']
                
                dataAsDF.at[doc_idx, 'cross-score'] = combined_score
                dataAsDF.at[doc_idx, 'dense-score'] = dense_score  # Store individual dense score
                dataAsDF.at[doc_idx, 'reranker-score'] = cross_score  # Store individual cross-encoder score
                
                if combined_score > dynamic_min_score:
                    dataAsDF.at[doc_idx, 'is_selected'] = True

    
def process_selection_event_day(dataAsDF, queryAsDF, sentence_retriver, cross_encoder, batch_size, event_no, device, top=200, dynamic_score=70, retriever_type="bm25") -> pd.DataFrame:
    
    # Process queries and get query responses
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        if retriever_type == "bm25":
            bm25, _ = build_bm25_index(dataAsDF)
            process_queries_bm25(queryAsDF, dataAsDF, bm25, cross_encoder, top)
        elif retriever_type == "dense":
            data_embeddings = get_embeddings(dataAsDF['search_text'].tolist(), sentence_retriver, batch_size, device).cpu().numpy()
            process_queries_dense(queryAsDF, data_embeddings, dataAsDF, sentence_retriver, cross_encoder, top, batch_size, device, dynamic_score=dynamic_score)
            
    # Get selected facts for the day based on cross-score and save them to a file with the scores
    selected_data = dataAsDF[dataAsDF['is_selected']].sort_values(by='cross-score', ascending=False).reset_index(drop=True)
    
    return selected_data.reset_index(drop=True)
