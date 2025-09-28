#!/usr/bin/env python3
"""
Simple launcher for the Fake News Detection System
"""
import os
import sys
import webbrowser
import time
from threading import Timer

def check_models():
    """Check if models exist, train if needed"""
    if not os.path.exists('models/fake_news_model.pkl'):
        print("🔧 Training basic model...")
        os.system('python train.py')
    
    if not os.path.exists('models/advanced_fake_news_model.pkl'):
        print("🔧 Training advanced model...")
        os.system('python train_advanced.py')

def open_browser():
    """Open browser after delay"""
    webbrowser.open('http://localhost:8082')

def main():
    print("🎯 FAKE NEWS DETECTION SYSTEM")
    print("=" * 40)
    
    # Check and train models if needed
    check_models()
    
    print("\n🌐 Starting web application...")
    print("📱 Browser will open automatically")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 40)
    
    # Open browser after 3 seconds
    Timer(3.0, open_browser).start()
    
    # Start Flask app
    try:
        from app import app
        app.run(debug=False, host='0.0.0.0', port=8082)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()