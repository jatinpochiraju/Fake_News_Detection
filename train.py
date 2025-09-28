#!/usr/bin/env python3
import pandas as pd
import joblib
import os
from src.data_processor import DataProcessor
from src.model import FakeNewsModel

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/sample_data.csv')
    print(f"Loaded {len(df)} samples")
    
    # Initialize processor and model
    processor = DataProcessor()
    model = FakeNewsModel(model_type='logistic')
    
    # Prepare data
    print("Preprocessing data...")
    df = processor.prepare_data(df)
    
    # Vectorize text
    X = processor.vectorize_text(df['text'])
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    
    # Train model
    print("Training model...")
    model.train(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    results = model.evaluate(X_test, y_test)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Save model and vectorizer
    print("Saving model...")
    os.makedirs('models', exist_ok=True)
    model.save_model('models/fake_news_model.pkl')
    joblib.dump(processor.vectorizer, 'models/vectorizer.pkl')
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()