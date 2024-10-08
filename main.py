import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

RAW_DATASET_PATH = 'datasets/news_articles.csv'
CLEANED_DATASET_PATH = 'datasets/news_articles_cleaned.csv'

def load_dataset(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def clean_dataset(dataset):
    dataset.dropna(axis=0, inplace=True)
    dataset.drop_duplicates(inplace=True)
    return dataset

def save_dataset(dataset, file_path):
    dataset.to_csv(file_path, index=False)

def calculate_source_statistics(dataset):
    source_counts = dataset.groupby(["site_url", "label"]).size().unstack(fill_value=0)
    
    total_news = source_counts["Real"] + source_counts["Fake"]
    source_counts["Percentage of Real News, %"] = (source_counts["Real"] / total_news) * 100
    source_counts["Percentage of Fake News, %"] = (source_counts["Fake"] / total_news) * 100
    
    return source_counts.sort_values(by="Percentage of Real News, %", ascending=False)

def print_top_sources(source_counts, top_n=10):
    print("Top 10 most reliable sources:")
    for source, row in source_counts.head(top_n).iterrows():
        print(f"News {source}: {row['Percentage of Fake News, %']:.1f}% of fake news")

    print("\nTop 10 least reliable sources:")
    for source, row in source_counts.tail(top_n).iterrows():
        print(f"News {source}: {row['Percentage of Fake News, %']:.1f}% of fake news")

def main():
    raw_news_dataset = load_dataset(RAW_DATASET_PATH)
    if raw_news_dataset is None:
        return

    cleaned_news_dataset = clean_dataset(raw_news_dataset)
    save_dataset(cleaned_news_dataset, CLEANED_DATASET_PATH)

    filtered_news_dataset = load_dataset(CLEANED_DATASET_PATH)
    if filtered_news_dataset is None:
        return

    sorted_source_counts = calculate_source_statistics(filtered_news_dataset)
    print_top_sources(sorted_source_counts)

if __name__ == "__main__":
    main()