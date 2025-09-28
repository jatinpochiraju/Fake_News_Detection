#!/usr/bin/env python3
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.advanced_processor import AdvancedDataProcessor
from src.advanced_model import AdvancedFakeNewsModel

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance"""
    importance_data = model.get_feature_importance(feature_names, top_n)
    
    if importance_data is None:
        print("Feature importance not available for this model type")
        return
    
    features, importances = zip(*importance_data)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(features)), importances)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, classes=['Real', 'Fake']):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load expanded data
    print("Loading expanded dataset...")
    df = pd.read_csv('data/balanced_dataset.csv')
    print(f"Loaded {len(df)} samples")
    print(f"Real news: {sum(df['label'] == 0)}, Fake news: {sum(df['label'] == 1)}")
    
    # Initialize advanced processor
    print("\nInitializing advanced processor...")
    processor = AdvancedDataProcessor()
    
    # Extract all features
    print("\nExtracting advanced features...")
    additional_features, cleaned_texts = processor.extract_all_features(df)
    
    print(f"Extracted {additional_features.shape[1]} additional features")
    print("Feature types:", additional_features.columns.tolist())
    
    # Prepare final feature matrix
    print("\nPreparing final feature matrix...")
    X = processor.prepare_final_features(cleaned_texts, additional_features, fit=True)
    y = df['label'].values
    
    print(f"Final feature matrix shape: {X.shape}")
    
    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = processor.split_data(X, y, test_size=0.2)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train different models and compare
    models_to_test = ['logistic', 'random_forest', 'xgboost', 'ensemble']
    results = {}
    
    for model_type in models_to_test:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} model...")
        print(f"{'='*50}")
        
        # Initialize and train model
        model = AdvancedFakeNewsModel(model_type=model_type)
        model.train(X_train, y_train)
        
        # Evaluate model
        print(f"\nEvaluating {model_type} model...")
        evaluation = model.evaluate(X_test, y_test)
        results[model_type] = evaluation
        
        print(f"Accuracy: {evaluation['accuracy']:.4f}")
        if evaluation['auc_score']:
            print(f"AUC Score: {evaluation['auc_score']:.4f}")
        
        print("\nClassification Report:")
        print(evaluation['classification_report'])
        
        # Cross-validation
        cv_results = model.cross_validate(X_train, y_train)
        print(f"\nCross-validation scores: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']*2:.4f})")
        
        # Save model if it's the best so far
        if model_type == 'ensemble':  # Save ensemble as the main model
            print(f"\nSaving {model_type} model...")
            os.makedirs('models', exist_ok=True)
            model.save_model('models/advanced_fake_news_model.pkl')
            joblib.dump(processor, 'models/advanced_processor.pkl')
            
            # Create feature names for visualization
            tfidf_features = [f'tfidf_{i}' for i in range(processor.tfidf_vectorizer.get_feature_names_out().shape[0])]
            additional_feature_names = additional_features.columns.tolist()
            all_feature_names = tfidf_features + additional_feature_names
            
            # Plot feature importance
            print("\nGenerating feature importance plot...")
            plot_feature_importance(model, all_feature_names)
            
            # Plot confusion matrix
            print("Generating confusion matrix plot...")
            plot_confusion_matrix(evaluation['confusion_matrix'])
    
    # Compare all models
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results.keys()],
        'AUC Score': [results[model]['auc_score'] if results[model]['auc_score'] else 0 for model in results.keys()]
    })
    
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
    best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Accuracy']
    
    print(f"\nBest performing model: {best_model.upper()} with accuracy: {best_accuracy:.4f}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("Files saved:")
    print("- models/advanced_fake_news_model.pkl")
    print("- models/advanced_processor.pkl")
    print("- models/feature_importance.png")
    print("- models/confusion_matrix.png")

if __name__ == "__main__":
    main()