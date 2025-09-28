import joblib
import os
import pandas as pd
import numpy as np
from .data_processor import DataProcessor

class RobustFakeNewsDetector:
    def __init__(self):
        self.basic_model = None
        self.basic_vectorizer = None
        self.advanced_model = None
        self.advanced_processor = None
        self.processor = DataProcessor()
        
        # Try to load models in order of preference
        self._load_models()
    
    def _load_models(self):
        """Load available models in order of preference"""
        # Try advanced model first (better for political content)
        try:
            if os.path.exists('models/advanced_fake_news_model.pkl') and os.path.exists('models/advanced_processor.pkl'):
                self.advanced_model = joblib.load('models/advanced_fake_news_model.pkl')
                self.advanced_processor = joblib.load('models/advanced_processor.pkl')
                print("✅ Advanced model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load advanced model: {e}")
        
        # Try basic model as backup
        try:
            if os.path.exists('models/fake_news_model.pkl') and os.path.exists('models/vectorizer.pkl'):
                self.basic_model = joblib.load('models/fake_news_model.pkl')
                self.basic_vectorizer = joblib.load('models/vectorizer.pkl')
                print("✅ Basic model loaded as backup")
        except Exception as e:
            print(f"⚠️ Failed to load basic model: {e}")
    
    def predict_single(self, text, source="unknown"):
        """Robust prediction with fallback options"""
        # Try advanced model first (better for political content)
        if self.advanced_model is not None and self.advanced_processor is not None:
            try:
                return self._predict_advanced(text, source)
            except Exception as e:
                print(f"Advanced model failed: {e}")
        
        # Fallback to basic model
        if self.basic_model is not None and self.basic_vectorizer is not None:
            try:
                return self._predict_basic(text, source)
            except Exception as e:
                print(f"Basic model failed: {e}")
        
        # Ultimate fallback - rule-based detection
        return self._predict_rule_based(text, source)
    
    def _predict_basic(self, text, source):
        """Basic model prediction"""
        # Clean and vectorize text
        cleaned_text = self.processor.clean_text(text)
        vectorized = self.basic_vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = self.basic_model.predict(vectorized)[0]
        probability = self.basic_model.predict_proba(vectorized)[0]
        
        confidence = max(probability)
        fake_prob = probability[1] if len(probability) > 1 else (1 - probability[0])
        real_prob = probability[0] if len(probability) > 1 else probability[0]
        
        return {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': confidence,
            'fake_probability': fake_prob,
            'real_probability': real_prob,
            'risk_level': self._get_risk_level(fake_prob),
            'features_summary': self._basic_feature_analysis(text, source),
            'model_used': 'Basic TF-IDF + Logistic Regression'
        }
    
    def _predict_advanced(self, text, source):
        """Advanced model prediction with error handling"""
        try:
            # Create temporary dataframe
            temp_df = pd.DataFrame({'text': [text], 'source': [source]})
            
            # Extract features
            additional_features, cleaned_texts = self.advanced_processor.extract_all_features(temp_df)
            
            # Prepare final features
            final_features = self.advanced_processor.prepare_final_features(
                cleaned_texts, additional_features, fit=False
            )
            
            # Make prediction
            prediction = self.advanced_model.predict(final_features)[0]
            probability = self.advanced_model.predict_proba(final_features)[0]
            
            confidence = max(probability)
            fake_prob = probability[1] if len(probability) > 1 else (1 - probability[0])
            real_prob = probability[0] if len(probability) > 1 else probability[0]
            
            return {
                'prediction': 'FAKE' if prediction == 1 else 'REAL',
                'confidence': confidence,
                'fake_probability': fake_prob,
                'real_probability': real_prob,
                'risk_level': self._get_risk_level(fake_prob),
                'features_summary': self._advanced_feature_analysis(additional_features.iloc[0]),
                'model_used': 'Advanced Ensemble Model'
            }
            
        except Exception as e:
            print(f"Advanced prediction failed: {e}")
            raise e
    
    def _predict_rule_based(self, text, source):
        """Rule-based fallback prediction"""
        fake_indicators = 0
        total_indicators = 0
        
        # Check for sensational language
        sensational_words = ['shocking', 'breaking', 'urgent', 'exposed', 'secret', 'hidden', 
                           'miracle', 'amazing', 'incredible', 'bombshell', 'exclusive']
        total_indicators += 1
        if any(word.lower() in text.lower() for word in sensational_words):
            fake_indicators += 1
        
        # Check for clickbait phrases
        clickbait_phrases = ['you won\'t believe', 'doctors hate', 'one simple trick', 
                           'this will shock you', 'what happens next']
        total_indicators += 1
        if any(phrase.lower() in text.lower() for phrase in clickbait_phrases):
            fake_indicators += 1
        
        # Check for excessive punctuation
        total_indicators += 1
        if text.count('!') > 3 or text.count('?') > 2:
            fake_indicators += 1
        
        # Check for all caps words
        words = text.split()
        caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
        total_indicators += 1
        if caps_words > len(words) * 0.1:  # More than 10% caps words
            fake_indicators += 1
        
        # Check source credibility
        credible_sources = ['reuters', 'bbc', 'cnn', 'nytimes', 'nature', 'science']
        suspicious_sources = ['blog', 'conspiracy', 'secret', 'exposed']
        
        total_indicators += 1
        if any(susp in source.lower() for susp in suspicious_sources):
            fake_indicators += 1
        elif any(cred in source.lower() for cred in credible_sources):
            fake_indicators -= 0.5  # Bonus for credible source
        
        # Calculate fake probability
        fake_prob = max(0.1, min(0.9, fake_indicators / total_indicators))
        
        return {
            'prediction': 'FAKE' if fake_prob > 0.5 else 'REAL',
            'confidence': abs(fake_prob - 0.5) + 0.5,
            'fake_probability': fake_prob,
            'real_probability': 1 - fake_prob,
            'risk_level': self._get_risk_level(fake_prob),
            'features_summary': {
                'analysis_method': 'Rule-based fallback',
                'sensational_language': 'Detected' if any(word.lower() in text.lower() for word in sensational_words) else 'Not detected',
                'source_assessment': 'Suspicious' if any(susp in source.lower() for susp in suspicious_sources) else 'Unknown'
            },
            'model_used': 'Rule-based Fallback'
        }
    
    def _get_risk_level(self, fake_probability):
        """Determine risk level based on fake probability"""
        if fake_probability < 0.3:
            return "LOW"
        elif fake_probability < 0.6:
            return "MEDIUM"
        elif fake_probability < 0.8:
            return "HIGH"
        else:
            return "VERY HIGH"
    
    def _basic_feature_analysis(self, text, source):
        """Basic feature analysis for simple model"""
        features = {}
        
        # Text length analysis
        word_count = len(text.split())
        if word_count < 50:
            features['text_length'] = "Very short"
        elif word_count < 200:
            features['text_length'] = "Short"
        elif word_count < 500:
            features['text_length'] = "Medium"
        else:
            features['text_length'] = "Long"
        
        # Punctuation analysis
        exclamation_count = text.count('!')
        if exclamation_count > 2:
            features['punctuation'] = "Excessive exclamations"
        elif exclamation_count > 0:
            features['punctuation'] = "Some exclamations"
        else:
            features['punctuation'] = "Normal punctuation"
        
        # Source analysis
        credible_indicators = ['reuters', 'bbc', 'cnn', 'nytimes', 'nature']
        if any(indicator in source.lower() for indicator in credible_indicators):
            features['source_type'] = "Appears credible"
        else:
            features['source_type'] = "Unknown reliability"
        
        return features
    
    def _advanced_feature_analysis(self, features):
        """Advanced feature analysis"""
        summary = {}
        
        try:
            # Sentiment analysis
            if 'sentiment_polarity' in features and not pd.isna(features['sentiment_polarity']):
                polarity = features['sentiment_polarity']
                if polarity > 0.1:
                    summary['sentiment'] = "Positive"
                elif polarity < -0.1:
                    summary['sentiment'] = "Negative"
                else:
                    summary['sentiment'] = "Neutral"
            
            # Source credibility
            if 'source_credible' in features and 'source_suspicious' in features:
                if features['source_credible'] == 1:
                    summary['source_type'] = "Credible"
                elif features['source_suspicious'] == 1:
                    summary['source_type'] = "Suspicious"
                else:
                    summary['source_type'] = "Unknown"
            
            # Writing style
            if 'style_exclamation_count' in features:
                exclamation_count = features['style_exclamation_count']
                if exclamation_count > 2:
                    summary['writing_style'] = "Sensational"
                else:
                    summary['writing_style'] = "Normal"
        
        except Exception as e:
            summary = {'analysis': 'Feature analysis unavailable', 'error': str(e)}
        
        return summary
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            'basic_model_available': self.basic_model is not None,
            'advanced_model_available': self.advanced_model is not None,
            'primary_model': 'Basic' if self.basic_model is not None else 'Advanced' if self.advanced_model is not None else 'Rule-based'
        }
        return info