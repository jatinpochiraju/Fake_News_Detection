#!/usr/bin/env python3
"""
Fix political bias in the fake news detection model
"""
import pandas as pd
import os

def combine_datasets():
    """Combine existing data with political training data"""
    print("🔧 FIXING POLITICAL BIAS IN MODEL")
    print("=" * 40)
    
    # Load existing data
    existing_df = pd.read_csv('data/expanded_data.csv')
    print(f"Existing data: {len(existing_df)} samples")
    
    # Load political data
    political_df = pd.read_csv('data/political_training_data.csv')
    print(f"Political data: {len(political_df)} samples")
    
    # Combine datasets
    combined_df = pd.concat([existing_df, political_df], ignore_index=True)
    
    # Save combined dataset
    combined_df.to_csv('data/balanced_dataset.csv', index=False)
    print(f"Combined data: {len(combined_df)} samples")
    
    # Show balance
    real_count = sum(combined_df['label'] == 0)
    fake_count = sum(combined_df['label'] == 1)
    print(f"Real news: {real_count}")
    print(f"Fake news: {fake_count}")
    print(f"Balance: {real_count/len(combined_df):.1%} real, {fake_count/len(combined_df):.1%} fake")
    
    return combined_df

def update_training_script():
    """Update training script to use balanced dataset"""
    
    # Read current training script
    with open('train_advanced.py', 'r') as f:
        content = f.read()
    
    # Replace dataset path
    content = content.replace(
        "df = pd.read_csv('data/expanded_data.csv')",
        "df = pd.read_csv('data/balanced_dataset.csv')"
    )
    
    # Write updated script
    with open('train_advanced.py', 'w') as f:
        f.write(content)
    
    print("✅ Updated training script to use balanced dataset")

def test_political_examples():
    """Test the model with political examples after retraining"""
    from src.robust_detector import RobustFakeNewsDetector
    
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
    
    print(f"\n🧪 TESTING POLITICAL BIAS FIX")
    print("=" * 40)
    
    detector = RobustFakeNewsDetector()
    correct = 0
    
    for i, case in enumerate(test_cases, 1):
        result = detector.predict_single(case['text'], case['source'])
        
        print(f"\n{i}. {case['text'][:50]}...")
        print(f"   Expected: {case['expected']}")
        print(f"   Predicted: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        
        if result['prediction'] == case['expected']:
            print("   ✅ CORRECT")
            correct += 1
        else:
            print("   ❌ INCORRECT - Still biased!")
    
    accuracy = correct / len(test_cases)
    print(f"\n🎯 Political Accuracy: {correct}/{len(test_cases)} ({accuracy:.1%})")
    
    if accuracy >= 0.8:
        print("🏆 Political bias successfully fixed!")
    else:
        print("⚠️  Political bias still exists - need more training")

def main():
    # Step 1: Combine datasets
    combined_df = combine_datasets()
    
    # Step 2: Update training script
    update_training_script()
    
    # Step 3: Retrain model
    print(f"\n🔄 RETRAINING MODEL WITH BALANCED DATA")
    print("=" * 40)
    os.system('python train_advanced.py')
    
    # Step 4: Test political examples
    test_political_examples()
    
    print(f"\n✅ POLITICAL BIAS FIX COMPLETE!")
    print("The model should now handle political news correctly.")

if __name__ == "__main__":
    main()