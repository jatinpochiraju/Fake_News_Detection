#!/usr/bin/env python3
"""
Test the political bias fix using the advanced model directly
"""
from src.advanced_processor import AdvancedDataProcessor
from src.advanced_model import AdvancedFakeNewsModel
import joblib
import pandas as pd

def test_advanced_model_directly():
    """Test the newly trained advanced model directly"""
    print("🧪 TESTING ADVANCED MODEL DIRECTLY")
    print("=" * 50)
    
    # Load the newly trained advanced model
    try:
        model = joblib.load('models/advanced_fake_news_model.pkl')
        processor = joblib.load('models/advanced_processor.pkl')
        print("✅ Advanced model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load advanced model: {e}")
        return
    
    test_cases = [
        {
            "text": "CNN projects Trump as winner of 2024 presidential race",
            "source": "CNN",
            "expected": "REAL"
        },
        {
            "text": "Associated Press calls election for Biden following vote count",
            "source": "Associated Press", 
            "expected": "REAL"
        },
        {
            "text": "Reuters reports final election results show decisive victory",
            "source": "Reuters",
            "expected": "REAL"
        },
        {
            "text": "BREAKING: Election was RIGGED by deep state operatives!",
            "source": "Conspiracy Blog",
            "expected": "FAKE"
        },
        {
            "text": "SHOCKING: Voting machines hacked to steal election!",
            "source": "Fake News Site",
            "expected": "FAKE"
        }
    ]
    
    correct = 0
    
    for i, case in enumerate(test_cases, 1):
        # Create dataframe for processing
        temp_df = pd.DataFrame({'text': [case['text']], 'source': [case['source']]})
        
        # Extract features
        additional_features, cleaned_texts = processor.extract_all_features(temp_df)
        
        # Prepare final features
        final_features = processor.prepare_final_features(cleaned_texts, additional_features, fit=False)
        
        # Make prediction
        prediction = model.predict(final_features)[0]
        probability = model.predict_proba(final_features)[0]
        
        result_label = 'FAKE' if prediction == 1 else 'REAL'
        confidence = max(probability)
        
        print(f"\n{i}. {case['text'][:50]}...")
        print(f"   Expected: {case['expected']}")
        print(f"   Predicted: {result_label}")
        print(f"   Confidence: {confidence:.1%}")
        
        if result_label == case['expected']:
            print("   ✅ CORRECT")
            correct += 1
        else:
            print("   ❌ INCORRECT")
    
    accuracy = correct / len(test_cases)
    print(f"\n🎯 Advanced Model Accuracy: {correct}/{len(test_cases)} ({accuracy:.1%})")
    
    if accuracy >= 0.8:
        print("🏆 Advanced model handles political news correctly!")
        return True
    else:
        print("⚠️  Advanced model still has political bias")
        return False

def update_robust_detector():
    """Update robust detector to prioritize advanced model"""
    
    print("\n🔧 UPDATING ROBUST DETECTOR")
    print("=" * 30)
    
    # Read current robust detector
    with open('src/robust_detector.py', 'r') as f:
        content = f.read()
    
    # Update the _load_models method to prioritize advanced model
    old_load_method = '''    def _load_models(self):
        """Load available models in order of preference"""
        # Try basic model first (most reliable)
        try:
            if os.path.exists('models/fake_news_model.pkl') and os.path.exists('models/vectorizer.pkl'):
                self.basic_model = joblib.load('models/fake_news_model.pkl')
                self.basic_vectorizer = joblib.load('models/vectorizer.pkl')
                print("✅ Basic model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load basic model: {e}")
        
        # Try advanced model as backup
        try:
            if os.path.exists('models/advanced_fake_news_model.pkl') and os.path.exists('models/advanced_processor.pkl'):
                self.advanced_model = joblib.load('models/advanced_fake_news_model.pkl')
                self.advanced_processor = joblib.load('models/advanced_processor.pkl')
                print("✅ Advanced model loaded as backup")
        except Exception as e:
            print(f"⚠️ Failed to load advanced model: {e}")'''
    
    new_load_method = '''    def _load_models(self):
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
            print(f"⚠️ Failed to load basic model: {e}")'''
    
    # Update predict_single method to try advanced model first
    old_predict_method = '''    def predict_single(self, text, source="unknown"):
        """Robust prediction with fallback options"""
        # Try basic model first (most reliable)
        if self.basic_model is not None and self.basic_vectorizer is not None:
            try:
                return self._predict_basic(text, source)
            except Exception as e:
                print(f"Basic model failed: {e}")
        
        # Fallback to advanced model
        if self.advanced_model is not None and self.advanced_processor is not None:
            try:
                return self._predict_advanced(text, source)
            except Exception as e:
                print(f"Advanced model failed: {e}")
        
        # Ultimate fallback - rule-based detection
        return self._predict_rule_based(text, source)'''
    
    new_predict_method = '''    def predict_single(self, text, source="unknown"):
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
        return self._predict_rule_based(text, source)'''
    
    # Replace methods
    content = content.replace(old_load_method, new_load_method)
    content = content.replace(old_predict_method, new_predict_method)
    
    # Write updated file
    with open('src/robust_detector.py', 'w') as f:
        f.write(content)
    
    print("✅ Updated robust detector to prioritize advanced model")

def main():
    # Test advanced model directly
    advanced_works = test_advanced_model_directly()
    
    if advanced_works:
        # Update robust detector to use advanced model first
        update_robust_detector()
        
        # Test with updated robust detector
        print("\n🧪 TESTING UPDATED ROBUST DETECTOR")
        print("=" * 40)
        
        from src.robust_detector import RobustFakeNewsDetector
        detector = RobustFakeNewsDetector()
        
        test_cases = [
            "CNN projects Trump as winner of 2024 presidential race",
            "Associated Press calls election for Biden",
            "BREAKING: Election was RIGGED by deep state!"
        ]
        
        for text in test_cases:
            result = detector.predict_single(text, "CNN")
            print(f"'{text[:40]}...' -> {result['prediction']} ({result['confidence']:.1%})")
        
        print("\n✅ POLITICAL BIAS FIX COMPLETE!")
        print("The system should now handle political news correctly.")
    else:
        print("\n❌ Advanced model still has issues. Need more training data.")

if __name__ == "__main__":
    main()