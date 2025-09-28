#!/usr/bin/env python3
import argparse
from src.detector import FakeNewsDetector

def main():
    parser = argparse.ArgumentParser(description='Detect fake news in text')
    parser.add_argument('--text', type=str, required=True, 
                       help='News text to analyze')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = FakeNewsDetector()
    
    try:
        # Make prediction
        result = detector.predict_single(args.text)
        
        print(f"\nNews Analysis Results:")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Fake Probability: {result['fake_probability']:.4f}")
        print(f"Real Probability: {result['real_probability']:.4f}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please run 'python train.py' first to train the model.")

if __name__ == "__main__":
    main()