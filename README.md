# 🎯 Fake News Detection System

A simple, effective fake news detection system with web interface and robust ML models.

## ✨ Features

- **Web Interface**: Easy-to-use browser interface
- **Text Analysis**: Paste news text for instant analysis
- **URL Analysis**: Extract and analyze content from news websites
- **Multiple Models**: Basic and advanced ML models with fallback
- **Risk Assessment**: 4-level risk scoring (LOW/MEDIUM/HIGH/VERY HIGH)
- **Source Analysis**: Automatic credibility assessment
- **Real-time Results**: Fast processing with confidence scores

## 🚀 Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. **Run the system:**
```bash
python run.py
```

3. **Open your browser to:** http://localhost:8082

## 📁 Project Structure

```
├── app.py                 # Web application
├── run.py                 # Simple launcher
├── test.py                # System testing
├── train.py               # Basic model training
├── train_advanced.py      # Advanced model training
├── requirements.txt       # Dependencies
├── src/
│   ├── robust_detector.py # Main detection engine
│   ├── data_processor.py  # Text preprocessing
│   ├── model.py          # ML models
│   ├── advanced_model.py # Advanced ML models
│   ├── advanced_processor.py # Advanced features
│   └── web_scraper.py    # URL content extraction
├── templates/
│   └── index.html        # Web interface
├── static/
│   ├── style.css         # Styling
│   └── script.js         # JavaScript
└── data/
    ├── sample_data.csv   # Basic training data
    └── expanded_data.csv # Extended training data
```

## 🧪 Testing

```bash
python test.py
```

## 💻 Usage Examples

### Python API
```python
from src.robust_detector import RobustFakeNewsDetector

detector = RobustFakeNewsDetector()
result = detector.predict_single(
    "Scientists discover breakthrough cancer treatment",
    "Harvard Medical School"
)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Risk Level: {result['risk_level']}")
```

### Web Interface
1. Run `python run.py`
2. Open http://localhost:8082
3. Enter text or URL
4. Get instant analysis with confidence scores

## 🎯 How It Works

1. **Text Processing**: Cleans and preprocesses input text
2. **Feature Extraction**: Analyzes sentiment, writing style, source credibility
3. **Model Prediction**: Uses ensemble of ML models for classification
4. **Risk Assessment**: Provides confidence scores and risk levels
5. **Fallback System**: Multiple models ensure reliability

## 📊 Performance

- **Accuracy**: 83%+ on test cases
- **Speed**: <0.01 seconds per analysis
- **Reliability**: Multi-model fallback system
- **Coverage**: Handles various news types and sources

## 🛠️ Customization

- **Add Training Data**: Edit `data/expanded_data.csv`
- **Retrain Models**: Run `python train_advanced.py`
- **Modify Features**: Edit `src/advanced_processor.py`
- **Adjust UI**: Modify `templates/index.html`

## 📋 Requirements

- Python 3.8+
- Flask for web interface
- scikit-learn for ML models
- spaCy for NLP features
- NLTK for text processing
- See `requirements.txt` for full list