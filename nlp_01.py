import pandas as pd
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import string

# Step 1: Download NLTK stopwords (only once)
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Step 2: Load the dataset
data = pd.read_csv("data/climate_nasa.csv")
print("📄 Total Comments:", len(data))

# Step 3: Drop missing text
data = data.dropna(subset=['text'])

# Step 4: Clean the text
def clean_text(text):
    text = text.lower()  # lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    words = text.split()  # tokenize
    words = [word for word in words if word not in stop_words]  # remove stopwords
    return " ".join(words)

data['clean_text'] = data['text'].apply(clean_text)

# Step 5: Combine all cleaned comments
all_words = " ".join(data['clean_text']).split()

# Step 6: Count top words
word_freq = Counter(all_words)
top_words = word_freq.most_common(20)

# Step 7: Display top 20 words
print("\n🔝 Top 20 Most Common Words:\n")
for word, freq in top_words:
    print(f"{word:<15} ➤ {freq} times")

# Step 8: Plot bar chart
words, freqs = zip(*top_words)
plt.figure(figsize=(10, 5))
plt.bar(words, freqs, color='skyblue')
plt.title("Top 20 Most Common Words in Climate Comments")
plt.xticks(rotation=45)
plt.ylabel("Frequency")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 9: Generate Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(all_words))
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("🌍 Word Cloud of Climate-Related Comments")
plt.show()
