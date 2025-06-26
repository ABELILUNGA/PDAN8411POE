import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Download NLTK resources (run this once)
nltk.download('stopwords')
nltk.download('wordnet')

# Load your data
df = pd.read_csv("rawdata.csv")

# Keep only relevant columns

df = df[['created_at', 'review_title', 'review_rating', 'review_content', 'business_name', 'industry_name']]
df.dropna(subset=['review_content', 'review_rating'], inplace=True)
df.drop_duplicates(subset=['review_content'], inplace=True)

# Set up text processing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

# Define cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # Remove URLs
    text = re.sub(r"\d+", "", text)                 # Remove digits
    tokens = tokenizer.tokenize(text)               # Tokenize words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

# Apply cleaning function
df["cleaned_review"] = df["review_content"].apply(clean_text)

# Save cleaned version (optional)
df.to_csv("cleaned_reviews.csv", index=False)

# Preview results
print(df[["review_content", "cleaned_review"]].head(3))


#EDA
import matplotlib.pyplot as plt

# Calculate number of words in each cleaned review
df['review_length'] = df['cleaned_review'].apply(lambda x: len(x.split()))

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(df['review_length'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Review Lengths (in Words)')
plt.xlabel('Number of Words')
plt.ylabel('Number of Reviews')
plt.grid(True)
plt.tight_layout()
plt.show()

print("Average length:", df['review_length'].mean())
print("Shortest review:", df['review_length'].min())
print("Longest review:", df['review_length'].max())


#Review Rating Distribution

import seaborn as sns
import matplotlib.pyplot as plt

# Convert rating to integers (if needed)
df['review_rating'] = df['review_rating'].astype(int)

# Count plot
plt.figure(figsize=(8, 5))
sns.countplot(x='review_rating', data=df, palette='Set2')
plt.title('Review Rating Distribution')
plt.xlabel('Review Rating (1 = Worst, 5 = Best)')
plt.ylabel('Number of Reviews')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Show value counts
print(df['review_rating'].value_counts().sort_index())


# Word Frequency & Word Cloud

from collections import Counter
from wordcloud import WordCloud

# Combine all cleaned reviews into one text blob
all_words = " ".join(df["cleaned_review"])

# Count word frequencies
word_counts = Counter(all_words.split())

# Top 20 most common words
top_words = word_counts.most_common(20)

# Convert to DataFrame for plotting
freq_df = pd.DataFrame(top_words, columns=["word", "count"])

# Plot top 20 frequent words
plt.figure(figsize=(10, 5))
sns.barplot(x="count", y="word", data=freq_df, palette="Blues_d")
plt.title("Top 20 Most Frequent Words in Reviews")
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.tight_layout()
plt.show()

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_words)

# Display word cloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of All Cleaned Reviews")
plt.tight_layout()
plt.show()


# Part 1: Feature Selection for Topic Modeling (LDA)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Use TF-IDF to convert cleaned text into feature matrix
tfidf = TfidfVectorizer(max_df=0.95, min_df=5, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['cleaned_review'])

# Save feature names (optional)
feature_names = tfidf.get_feature_names_out()
print("Shape of TF-IDF matrix:", tfidf_matrix.shape)


# Part 2: Feature Selection for Topic Modeling (LDA)
# Map review ratings to sentiment labels
def map_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

df['sentiment'] = df['review_rating'].apply(map_sentiment)

# Check distribution
print(df['sentiment'].value_counts())

#Temporal Analysis (Trends Over Time)

import matplotlib.pyplot as plt
import seaborn as sns

# Convert to datetime if not already
df['created_at'] = pd.to_datetime(df['created_at'])

# Create month-year column
df['month_year'] = df['created_at'].dt.to_period('M').astype(str)

# Review counts over time
plt.figure(figsize=(12, 5))
df.groupby('month_year').size().plot(kind='line', marker='o', color='royalblue')
plt.title('Total Reviews Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Average sentiment polarity over time
sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
df['sentiment_score'] = df['sentiment'].map(sentiment_map)

# Average sentiment by month
monthly_sentiment = df.groupby('month_year')['sentiment_score'].mean()

plt.figure(figsize=(12, 5))
monthly_sentiment.plot(marker='o', color='darkgreen')
plt.title('Average Sentiment Score Over Time')
plt.xlabel('Month')
plt.ylabel('Sentiment Score (-1 = Negative, +1 = Positive)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Calculate correlation
correlation = df['review_rating'].astype(float).corr(df['review_length'])

# Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='review_rating', y='review_length', alpha=0.3)
plt.title(f'Review Rating vs Review Length (Correlation = {correlation:.2f})')
plt.xlabel('Review Rating')
plt.ylabel('Review Length (in Words)')
plt.tight_layout()
plt.show()


#Model training and evaluation form here 

if __name__ == "__main__":
    import gensim
    from gensim import corpora
    from gensim.models import CoherenceModel
    import pyLDAvis.gensim_models
    import nltk
    nltk.download('stopwords')

    # Tokenize
    tokenized_reviews = df['cleaned_review'].apply(lambda x: x.split())

    # Dictionary & Corpus
    dictionary = corpora.Dictionary(tokenized_reviews)
    dictionary.filter_extremes(no_below=5, no_above=0.95)
    corpus = [dictionary.doc2bow(text) for text in tokenized_reviews]

    # Train LDA
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=5,
        random_state=42,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )

    # Print topics
    topics = lda_model.print_topics()
    for topic_num, topic_words in topics:
        print(f"Topic {topic_num}: {topic_words}")

    # Coherence score
    coherence_model = CoherenceModel(model=lda_model, texts=tokenized_reviews, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(f"\nCoherence Score: {coherence_score:.4f}")

    # Save interactive HTML
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'lda_topics.html')

# ================================
# 4.3.2 Sentiment Classification
# ================================

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# TF-IDF Vectorization (re-run to keep it scoped here)
vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words='english')
X = vectorizer.fit_transform(df['cleaned_review'])

# Target (Sentiment column created earlier)
y = df['sentiment']

# Split into training and testing sets (Stratified for balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix: Sentiment Classifier")
plt.tight_layout()
plt.show()
