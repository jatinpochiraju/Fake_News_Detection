from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib
import os
import numpy as np

class AdvancedFakeNewsModel:
    def __init__(self, model_type='ensemble'):
        """Initialize advanced model with specified type"""
        self.model_type = model_type
        
        if model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        elif model_type == 'xgboost':
            self.model = xgb.XGBClassifier(random_state=42, n_estimators=200, max_depth=6, learning_rate=0.1)
        elif model_type == 'svm':
            self.model = SVC(random_state=42, probability=True, kernel='rbf', C=1.0)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(random_state=42, n_estimators=200, learning_rate=0.1)
        elif model_type == 'ensemble':
            # Create ensemble of multiple models
            lr = LogisticRegression(random_state=42, max_iter=1000)
            rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
            xgb_model = xgb.XGBClassifier(random_state=42, n_estimators=100, max_depth=4)
            gb = GradientBoostingClassifier(random_state=42, n_estimators=100)
            
            self.model = VotingClassifier(
                estimators=[
                    ('lr', lr),
                    ('rf', rf),
                    ('xgb', xgb_model),
                    ('gb', gb)
                ],
                voting='soft'  # Use probability averaging
            )
        else:
            raise ValueError("Model type must be one of: 'logistic', 'random_forest', 'xgboost', 'svm', 'gradient_boosting', 'ensemble'")
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Comprehensive model evaluation"""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        
        # ROC AUC score
        try:
            auc_score = roc_auc_score(y_test, probabilities[:, 1])
        except:
            auc_score = None
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            feature_importance = np.abs(self.model.coef_[0])
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': report,
            'confusion_matrix': cm,
            'feature_importance': feature_importance
        }
    
    def get_feature_importance(self, feature_names=None, top_n=20):
        """Get top feature importances"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Get top features
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [(feature_names[i], importances[i]) for i in indices]
        
        return top_features
    
    def save_model(self, filepath):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        from sklearn.model_selection import cross_val_score
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        }