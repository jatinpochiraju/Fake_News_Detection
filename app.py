#!/usr/bin/env python3
from flask import Flask, render_template, request, jsonify, flash
import os
import time
from src.robust_detector import RobustFakeNewsDetector
from src.web_scraper import NewsWebScraper
import traceback

app = Flask(__name__)
app.secret_key = 'fake_news_detector_secret_key_2024'

# Initialize components
detector = RobustFakeNewsDetector()
scraper = NewsWebScraper()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze text or URL for fake news"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        input_type = data.get('type', 'text')
        content = data.get('content', '').strip()
        
        if not content:
            return jsonify({'error': 'No content provided'}), 400
        
        start_time = time.time()
        
        if input_type == 'url':
            # Extract content from URL
            article_data, error = scraper.extract_from_url(content)
            
            if error:
                return jsonify({'error': f'Failed to extract content: {error}'}), 400
            
            # Analyze the extracted content
            result = detector.predict_single(article_data['text'], article_data['source'])
            
            # Add article metadata
            result['article'] = {
                'title': article_data['title'],
                'source': article_data['source'],
                'url': content,
                'authors': article_data.get('authors', []),
                'publish_date': article_data.get('publish_date'),
                'word_count': len(article_data['text'].split())
            }
            
        else:
            # Analyze direct text input
            source = data.get('source', 'User Input')
            result = detector.predict_single(content, source)
            
            result['article'] = {
                'title': 'Direct Text Input',
                'source': source,
                'word_count': len(content.split())
            }
        
        # Add processing time
        result['processing_time'] = round(time.time() - start_time, 2)
        
        # Add interpretation
        result['interpretation'] = _get_interpretation(result)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in analyze: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/quick-analyze', methods=['POST'])
def quick_analyze():
    """Quick analysis using basic model"""
    try:
        data = request.get_json()
        content = data.get('content', '').strip()
        
        if not content:
            return jsonify({'error': 'No content provided'}), 400
        
        start_time = time.time()
        
        # Try to use basic detector first
        try:
            import joblib
            import os
            from src.data_processor import DataProcessor
            
            if os.path.exists('models/fake_news_model.pkl') and os.path.exists('models/vectorizer.pkl'):
                # Load basic model
                model = joblib.load('models/fake_news_model.pkl')
                vectorizer = joblib.load('models/vectorizer.pkl')
                
                # Basic processing
                processor = DataProcessor()
                cleaned_text = processor.clean_text(content)
                vectorized = vectorizer.transform([cleaned_text])
                
                # Predict
                prediction = model.predict(vectorized)[0]
                probability = model.predict_proba(vectorized)[0]
                
                result = {
                    'prediction': 'FAKE' if prediction == 1 else 'REAL',
                    'confidence': max(probability),
                    'fake_probability': probability[1] if len(probability) > 1 else (1 - probability[0]),
                    'real_probability': probability[0] if len(probability) > 1 else probability[0],
                    'risk_level': 'HIGH' if max(probability) > 0.7 and prediction == 1 else 'MEDIUM' if prediction == 1 else 'LOW'
                }
            else:
                raise FileNotFoundError("Basic model not found")
                
        except Exception as basic_error:
            print(f"Basic model failed: {basic_error}")
            # Fallback to advanced detector
            result = detector.predict_single(content, "Quick Analysis")
        
        result['processing_time'] = round(time.time() - start_time, 2)
        result['mode'] = 'quick'
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Quick analysis error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Quick analysis failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    model_info = detector.get_model_info()
    return jsonify({
        'status': 'healthy', 
        'model_info': model_info,
        'primary_model': model_info['primary_model']
    })

def _get_interpretation(result):
    """Get human-readable interpretation of results"""
    prediction = result['prediction']
    confidence = result['confidence']
    risk_level = result['risk_level']
    fake_prob = result['fake_probability']
    
    interpretations = []
    
    # Main prediction
    if prediction == 'FAKE':
        interpretations.append(f"This content is classified as FAKE NEWS with {confidence:.1%} confidence.")
    else:
        interpretations.append(f"This content appears to be LEGITIMATE NEWS with {confidence:.1%} confidence.")
    
    # Risk assessment
    if risk_level == 'VERY HIGH':
        interpretations.append("⚠️ VERY HIGH RISK: Strong indicators suggest this is fake news. Avoid sharing.")
    elif risk_level == 'HIGH':
        interpretations.append("🚨 HIGH RISK: Multiple red flags detected. Verify with credible sources before sharing.")
    elif risk_level == 'MEDIUM':
        interpretations.append("🔍 MEDIUM RISK: Some suspicious indicators. Cross-check with other reliable sources.")
    else:
        interpretations.append("✅ LOW RISK: Content appears legitimate based on our analysis.")
    
    # Feature insights
    features = result.get('features_summary', {})
    
    if 'source_type' in features:
        if features['source_type'] == 'Credible':
            interpretations.append("✓ Source appears to be from a credible news outlet.")
        elif features['source_type'] == 'Suspicious':
            interpretations.append("⚠️ Source appears to be suspicious or unreliable.")
    
    if 'sentiment' in features:
        sentiment = features['sentiment']
        if sentiment in ['Very Positive', 'Very Negative']:
            interpretations.append(f"📊 Content shows {sentiment.lower()} sentiment, which may indicate bias.")
    
    if 'writing_style' in features:
        style = features['writing_style']
        if 'Sensational' in style:
            interpretations.append("📝 Writing style appears sensational with excessive punctuation.")
        elif 'Aggressive' in style:
            interpretations.append("📝 Writing style uses excessive capital letters.")
    
    return interpretations

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=False, host='0.0.0.0', port=8082)