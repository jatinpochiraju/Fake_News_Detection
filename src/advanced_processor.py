import pandas as pd
import numpy as np
import re
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import spacy

class AdvancedDataProcessor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.count_vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        # Load spaCy model (install with: python -m spacy download en_core_web_sm)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def clean_text(self, text):
        """Enhanced text cleaning"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep some punctuation
        text = re.sub(r'[^a-zA-Z\s\.\!\?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_writing_style_features(self, text):
        """Extract writing style features"""
        features = {}
        
        if not text or pd.isna(text):
            return {f'style_{k}': 0 for k in ['avg_word_len', 'exclamation_count', 'question_count', 
                                             'caps_ratio', 'sentence_count', 'avg_sentence_len']}
        
        # Average word length
        words = text.split()
        features['style_avg_word_len'] = np.mean([len(word) for word in words]) if words else 0
        
        # Punctuation counts
        features['style_exclamation_count'] = text.count('!')
        features['style_question_count'] = text.count('?')
        
        # Capital letters ratio
        if len(text) > 0:
            features['style_caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
        else:
            features['style_caps_ratio'] = 0
        
        # Sentence statistics
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        features['style_sentence_count'] = len(sentences)
        features['style_avg_sentence_len'] = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        
        return features
    
    def extract_sentiment_features(self, text):
        """Extract sentiment features using TextBlob"""
        features = {}
        
        if not text or pd.isna(text):
            return {'sentiment_polarity': 0, 'sentiment_subjectivity': 0}
        
        blob = TextBlob(text)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        
        return features
    
    def extract_named_entities(self, text):
        """Extract named entity features using spaCy"""
        features = {}
        
        if not text or pd.isna(text) or self.nlp is None:
            return {f'ner_{ent_type}': 0 for ent_type in ['PERSON', 'ORG', 'GPE', 'MONEY', 'DATE']}
        
        doc = self.nlp(text)
        ent_counts = {}
        
        for ent in doc.ents:
            ent_counts[ent.label_] = ent_counts.get(ent.label_, 0) + 1
        
        # Focus on key entity types
        key_entities = ['PERSON', 'ORG', 'GPE', 'MONEY', 'DATE']
        for ent_type in key_entities:
            features[f'ner_{ent_type}'] = ent_counts.get(ent_type, 0)
        
        return features
    
    def extract_source_credibility_features(self, source):
        """Extract source credibility features"""
        features = {}
        
        if pd.isna(source):
            source = "unknown"
        
        source = source.lower()
        
        # Credible source indicators
        credible_sources = ['reuters', 'bbc', 'cnn', 'nytimes', 'washingtonpost', 'guardian', 
                           'associated press', 'npr', 'pbs', 'wsj', 'financial times',
                           'nature', 'science', 'medical journal', 'university', 'nasa',
                           'who', 'government', 'academic']
        
        features['source_credible'] = int(any(cred in source for cred in credible_sources))
        
        # Suspicious source indicators
        suspicious_sources = ['blog', 'unknown', 'scam', 'conspiracy', 'tabloid', 'clickbait',
                             'fake', 'parody', 'satirical', 'ad network']
        
        features['source_suspicious'] = int(any(susp in source for susp in suspicious_sources))
        
        return features
    
    def extract_all_features(self, df, text_column='text', source_column='source'):
        """Extract all features from the dataset"""
        feature_dfs = []
        
        # Clean text
        df[text_column] = df[text_column].apply(self.clean_text)
        
        # Extract various features
        print("Extracting writing style features...")
        style_features = df[text_column].apply(self.extract_writing_style_features)
        style_df = pd.DataFrame(style_features.tolist())
        feature_dfs.append(style_df)
        
        print("Extracting sentiment features...")
        sentiment_features = df[text_column].apply(self.extract_sentiment_features)
        sentiment_df = pd.DataFrame(sentiment_features.tolist())
        feature_dfs.append(sentiment_df)
        
        if source_column in df.columns:
            print("Extracting source credibility features...")
            source_features = df[source_column].apply(self.extract_source_credibility_features)
            source_df = pd.DataFrame(source_features.tolist())
            feature_dfs.append(source_df)
        
        if self.nlp is not None:
            print("Extracting named entity features...")
            ner_features = df[text_column].apply(self.extract_named_entities)
            ner_df = pd.DataFrame(ner_features.tolist())
            feature_dfs.append(ner_df)
        
        # Combine all features
        combined_features = pd.concat(feature_dfs, axis=1)
        
        return combined_features, df[text_column]
    
    def vectorize_text(self, texts, fit=True, method='tfidf'):
        """Convert text to vectors"""
        if method == 'tfidf':
            if fit:
                return self.tfidf_vectorizer.fit_transform(texts)
            else:
                return self.tfidf_vectorizer.transform(texts)
        elif method == 'count':
            if fit:
                return self.count_vectorizer.fit_transform(texts)
            else:
                return self.count_vectorizer.transform(texts)
    
    def prepare_final_features(self, texts, additional_features, fit=True):
        """Combine text vectors with additional features"""
        # Get TF-IDF vectors
        text_vectors = self.vectorize_text(texts, fit=fit)
        
        # Convert to dense array and combine with additional features
        text_dense = text_vectors.toarray()
        additional_dense = additional_features.values
        
        # Combine features
        final_features = np.hstack([text_dense, additional_dense])
        
        return final_features
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)