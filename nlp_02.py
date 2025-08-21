import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Step 1: Load cleaned dataset
data = pd.read_csv("data/climate_nasa.csv")
data = data.dropna(subset=['text'])

# Step 2: Clean text like in Division 1
def clean_text(text):
    import string
    from nltk.corpus import stopwords
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# Optional: Clean again if needed
data['clean_text'] = data['text'].apply(clean_text)

# Step 3: Sentiment Analysis using TextBlob
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

data['Sentiment'] = data['clean_text'].apply(get_sentiment)

# Step 4: Count sentiments
sentiment_counts = data['Sentiment'].value_counts()
print("\n📊 Sentiment Summary:")
print(sentiment_counts)

# Step 5: Plot the sentiment distribution
colors = ['green', 'red', 'gray']
sentiment_counts.plot(kind='bar', color=colors, title="Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Comments")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Optional: Pie Chart
plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title("Sentiment Breakdown of Climate Comments")
plt.show()
