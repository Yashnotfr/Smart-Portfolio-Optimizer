#!/usr/bin/env python3
"""
Run script for Smart Portfolio Optimizer Dashboard

This script starts the Flask web application for the portfolio optimizer.
"""

import sys
import os
import webbrowser
import threading
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def open_browser():
    """Open web browser after a short delay."""
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

def main():
    """Main function to run the dashboard."""
    print("🚀 Starting Smart Portfolio Optimizer Dashboard...")
    print("=" * 50)
    
    try:
        # Import and run the Flask app
        from dashboard.app import app
        
        print("✅ Dashboard initialized successfully!")
        print("📊 Opening web browser...")
        print("🌐 Dashboard will be available at: http://localhost:5000")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Start browser in a separate thread
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run the Flask app
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"❌ Error importing dashboard: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
