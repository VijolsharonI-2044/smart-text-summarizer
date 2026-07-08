import re
import heapq
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# One-time downloads
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

def summarize(text, num_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    stop_words = set(stopwords.words('english'))
    word_freq = defaultdict(int)

    # Clean text and build word frequency table
    for word in word_tokenize(text.lower()):
        word = re.sub(r'[^\w\s]', '', word)
        if word and word not in stop_words:
            word_freq[word] += 1

    max_freq = max(word_freq.values())
    for word in word_freq:
        word_freq[word] /= max_freq  # normalize

    # Score each sentence by summing normalized word frequencies
    sentence_scores = defaultdict(float)
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            word = re.sub(r'[^\w\s]', '', word)
            if word in word_freq:
                sentence_scores[i] += word_freq[word]
        # Penalize very long sentences slightly (avoid bias toward long ones)
        sentence_scores[i] /= (len(word_tokenize(sentence)) ** 0.3 + 1)

    # Pick top-N sentences, keep original order
    top_idx = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    top_idx.sort()

    return ' '.join(sentences[i] for i in top_idx)


if __name__ == "__main__":
    text = """Paste your long text here. It can be several paragraphs.
    The summarizer will pick out the most important sentences based on
    word frequency scoring, which is a simple but effective extractive
    summarization technique used widely before transformer models existed."""

    print(summarize(text, num_sentences=3))
