import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import chi2_contingency

nltk.download("vader_lexicon")

RAW_DATASET_PATH = 'datasets/news_articles.csv'
CLEANED_DATASET_PATH = 'datasets/news_articles_cleaned.csv'

STOP_WORDS = set(stopwords.words('english'))


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

def process_words(text, stop_words):
    words = word_tokenize(text)
    return [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]


def update_counters(row, title_counter, text_counter, stop_words):
    title_words = process_words(row["title"], stop_words)
    text_words = process_words(row["text"], stop_words)

    if row["label"] == "Fake":
        title_counter.update(title_words)
        text_counter.update(text_words)

def print_top_keywords(counter, description, top_n=10):
    top_keywords = counter.most_common(top_n)
    print(f"\nTop {top_n} most common words in {description}:")
    for word, count in top_keywords:
        print(f"{word}: {count} times")

def calculate_average_lengths(dataset):
    """Calculate average title and text lengths for real and fake news."""
    dataset["title_length"] = dataset["title"].apply(len)
    dataset["text_length"] = dataset["text"].apply(len)

    real_news = dataset[dataset["label"] == "Real"]
    fake_news = dataset[dataset["label"] == "Fake"]

    avg_real_title_length = real_news["title_length"].mean()
    avg_fake_title_length = fake_news["title_length"].mean()
    avg_real_text_length = real_news["text_length"].mean()
    avg_fake_text_length = fake_news["text_length"].mean()

    return avg_real_title_length, avg_fake_title_length, avg_real_text_length, avg_fake_text_length

def print_average_lengths(avg_real_title_length, avg_fake_title_length, avg_real_text_length, avg_fake_text_length):
    """Print average title and text lengths for real and fake news."""
    print(f"\nAverage title length for real news: {avg_real_title_length:.1f} characters")
    print(f"Average title length for fake news: {avg_fake_title_length:.1f} characters")
    print(f"Average text length for real news: {avg_real_text_length:.1f} characters")
    print(f"Average text length for fake news: {avg_fake_text_length:.1f} characters")

def plot_average_lengths(labels, lengths, title, ylabel):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, lengths, color=["green", "red"])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.show()

def detect_sensationalism(text):
    sensational_words = ["shocking", "amazing", "unbelievable", "miracle", "incredible"]
    for keyword in sensational_words:
        if re.search(r"\b" + keyword + r"\b", text, re.IGNORECASE):
            return True
        return False
    
def perform_chi_square_test(dataset):
    dataset["sensationalism"] = dataset["text"].apply(detect_sensationalism)
    contingency_table = pd.crosstab(dataset["sensationalism"], dataset["label"])
    print(contingency_table)

    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-square statistic: {chi2:.4f}")
    print(f"p-value: {p:.4f}")

    alpha = 0.05
    if p <= alpha:
        print("\nReject the null hypothesis: There is a significant relationship between sensationalism and fake news.")
    else:
        print("\nFail to reject the null hypothesis: There is no significant relationship between sensationalism and fake news.")

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    if sentiment_score["compound"] >= 0.05:
        return "Positive"
    elif sentiment_score["compound"] <= -0.05:
        return "Negative"
    else:
        return "Neutral"


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

    title_counter = Counter()
    text_counter = Counter()

    for _, row in filtered_news_dataset.iterrows():
        update_counters(row, title_counter, text_counter, STOP_WORDS)

    print_top_keywords(title_counter, "fake news titles")
    print_top_keywords(text_counter, "fake news text")

    avg_real_title_length, avg_fake_title_length, avg_real_text_length, avg_fake_text_length = calculate_average_lengths(filtered_news_dataset)
    print_average_lengths(avg_real_title_length, avg_fake_title_length, avg_real_text_length, avg_fake_text_length)

    title_labels = ["Real Title", "Fake Title"]
    title_lengths = [avg_real_title_length, avg_fake_title_length]
    plot_average_lengths(title_labels, title_lengths, "Average Title Lengths for Real and Fake News", "Average Length (characters)")

    text_labels = ["Real Text", "Fake Text"]
    text_lengths = [avg_real_text_length, avg_fake_text_length]
    plot_average_lengths(text_labels, text_lengths, "Average Text Lengths for Real and Fake News", "Average Length (characters)")

    perform_chi_square_test(filtered_news_dataset)

    filtered_news_dataset["sentiment"] = filtered_news_dataset["text"].apply(analyze_sentiment)
    print("\nSentiment analysis results:")
    print(filtered_news_dataset[["text","sentiment"]].head())

if __name__ == "__main__":
    main()