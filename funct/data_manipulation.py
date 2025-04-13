import re
import emoji
import pandas as pd
from funct.utils import load_query_data

def get_emoji_regexp():
    # from emoji library, it was deprecated in the newer versions
    
    # Sort emoji by length to make sure multi-character emojis are
    # matched first
    emojis = sorted(emoji.EMOJI_DATA, key=len, reverse=True)
    pattern = '(' + '|'.join(re.escape(u) for u in emojis) + ')'
    return re.compile(pattern)

def sanitize_data(text: str) -> str:
    """ Sanitize the input text by removing unwanted characters and formatting."""
    
    # Remove URLs, hashtags, and mentions and replace with space
    text = re.sub(r"http\S+|www\S+|https\S+|@|#", " ", text)
    
    # Remove extra \n and \t characters and extra spaces
    text = re.sub(r"\n|\t", " ", text)
    text = re.sub(r'\s+', " ", text).strip()
    
    # Remove all emojis
    text = re.sub(get_emoji_regexp(), "", text)
    
    return text

def reformulate_info(info: str) -> str:
    """ Reformulate the info string based on the loaded query data."""
    
    query_data = load_query_data()
    return query_data["info"].get(info, info)

def reformulate_info_simpler(info: str) -> str:
    """ Reformulate the info string based on the loaded query data in a simpler format."""
    
    query_data = load_query_data()
    return query_data["info_simpler"].get(info, info)

def reformulate_query(query, event=None, simple=False):
    event = reformulate_info(event) if not simple else reformulate_info_simpler(event)
    query_data = load_query_data()
    return query_data["queries"].get(query, query).format(event=event)

def add_default_columns(data:pd.DataFrame):
    """ Add default columns to the DataFrame."""
    
    data['is_selected'] = False
    data['cross-score'] = 0.0
    data['query'] = ""
    data['answer'] = ""
    data['answer-score'] = 0.0
    data['topic'] = -1
    data['topic_score'] = 0.0
    data['topic_words'] = ""
    data['embedding'] = None

def remove_default_columns(data: pd.DataFrame):
    """ Remove default columns from the DataFrame."""
    
    data.drop(columns=['is_selected', 'cross-score', 'query', 'answer', 'answer-score', 'topic', 'topic_score', 'topic_words', 'embedding'], inplace=True)
    
def prepare_queries(queryAsDF: pd.DataFrame) -> pd.DataFrame:
    """ Prepare the queries by adding default columns, add other queries and reformulating them."""
    
    query_data = load_query_data(json_path="./json_files/query_added.json")
    general_queries = query_data["general_queries"]
    event_specific_queries = query_data["event_specific_queries"]

    expanded_queries = set()  # Avoid duplicates

    for _, row in queryAsDF.iterrows():
        event_id = row["event_id"]
        event_type = row.get("event_type", None)  # Get event type if available
        event_title = row["event_title"]

        # **Add General Queries**
        for query in general_queries:
            expanded_queries.add((
                f"{event_id}_general_{query}",
                query.format(event=event_title),
                event_id,
                event_title,
                row["event_description"]
            ))

        # **Add Event-Specific Queries**
        if event_type in event_specific_queries:
            for query in event_specific_queries[event_type]:
                expanded_queries.add((
                    f"{event_id}_{event_type}_{query}",
                    query.format(event=event_title),
                    event_id,
                    event_title,
                    row["event_description"]
                ))

    # Convert set to DataFrame
    expanded_query_df = pd.DataFrame(expanded_queries, columns=["query_id", "text", "event_id", "event_title", "event_description"])

    # Combine with the original dataset
    queryAsDF = pd.concat([queryAsDF, expanded_query_df], ignore_index=True)

    # Reformulate queries
    queryAsDF["revised_text"] = queryAsDF.apply(
        lambda row: reformulate_query(row["text"], row["event_title"], simple=True),
        axis=1
    )

    queryAsDF["long_query"] = queryAsDF.apply(
        lambda row: reformulate_query(row["text"], row["event_title"], simple=False),
        axis=1
    )

    return queryAsDF