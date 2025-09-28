#!/usr/bin/env python3
"""
Simple test script for the Fake News Detection System
"""
from src.robust_detector import RobustFakeNewsDetector

def main():
    print("🧪 TESTING FAKE NEWS DETECTION SYSTEM")
    print("=" * 50)
    
    # Test cases
    tests = [
        {
            "text": "Scientists at Harvard discover breakthrough cancer treatment with 95% success rate in clinical trials",
            "source": "Harvard Medical School",
            "expected": "REAL"
        },
        {
            "text": "SHOCKING: Government hiding miracle cure! This one simple trick doctors don't want you to know!",
            "source": "Health Blog",
            "expected": "FAKE"
        },
        {
            "text": "Federal Reserve announces 0.25% interest rate increase following economic data review",
            "source": "Reuters",
            "expected": "REAL"
        },
        {
            "text": "URGENT: Stock market will crash tomorrow! Secret insider information revealed!",
            "source": "Investment Scam",
            "expected": "FAKE"
        }
    ]
    
    # Initialize detector
    detector = RobustFakeNewsDetector()
    
    # Run tests
    correct = 0
    for i, test in enumerate(tests, 1):
        print(f"\n📰 Test {i}: {test['text'][:60]}...")
        
        result = detector.predict_single(test['text'], test['source'])
        
        print(f"   Expected: {test['expected']}")
        print(f"   Predicted: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Risk: {result['risk_level']}")
        
        if result['prediction'] == test['expected']:
            print("   ✅ CORRECT")
            correct += 1
        else:
            print("   ❌ INCORRECT")
    
    # Results
    accuracy = correct / len(tests)
    print(f"\n🎯 RESULTS: {correct}/{len(tests)} correct ({accuracy:.1%})")
    
    if accuracy >= 0.75:
        print("🏆 System working well!")
    else:
        print("⚠️  System needs improvement")

if __name__ == "__main__":
    main()