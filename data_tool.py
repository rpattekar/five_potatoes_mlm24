import re
import jieba
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup
from textblob import TextBlob
import emoji
from collections import Counter
import dateparser
import spacy

# Original text
text = "Hello world! This is a sample text with a URL https://example.com and an email: example@example.com. 今天是2023年9月29日，今天天气很好！ 😊"

# 1. Remove HTML tags
text = BeautifulSoup(text, "html.parser").get_text()

# 2. Remove URLs and email addresses
text = re.sub(r'http\S+|www\S+|@\S+', '', text)

# 3. Convert to lowercase
text = text.lower()

# 4. Remove punctuation and special characters
text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\s]', '', text)  # Keep English, Chinese characters, and numbers

# 5. Tokenization (English using NLTK, Chinese using Jieba)
tokens = word_tokenize(text)
tokens += jieba.lcut(text)

# 6. Remove stopwords
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words]

# 7. Stemming and Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stemmed_words = [stemmer.stem(word) for word in tokens]
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

# 8. Remove numbers
tokens = [word for word in lemmatized_words if not word.isdigit()]

# 9. Remove extra spaces
text = " ".join(tokens)

# 10. Correct spelling errors
text = str(TextBlob(text).correct())

# 11. Handle emojis
text = emoji.demojize(text)

# 12. Normalize (e.g., dates)
normalized_date = dateparser.parse("2023年9月29日")

# 13. Handle low-frequency and high-frequency words
word_counts = Counter(tokens)
low_freq_words = [word for word, count in word_counts.items() if count < 2]
tokens = [word for word in tokens if word not in low_freq_words]

# 14. Further NLP processing using spaCy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
tokens = [token.lemma_ for token in doc if not token.is_stop]

# Cleaned text
cleaned_text = " ".join(tokens)
print(cleaned_text)
