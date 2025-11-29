import uvicorn
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

if __name__ == "__main__":
    print("=" * 50)
    print("🏋️  AI Exercise Trainer - Web Server")
    print("=" * 50)
    print("\n📍 Server starting at: http://localhost:8000")
    print("📹 Open your browser and navigate to the URL above")
    print("\n⚠️  Make sure you have:")
    print("   - Webcam connected (for live analysis)")
    print("   - GOOGLE_API_KEY in .env file (for AI feedback)")
    print("\n🛑 Press CTRL+C to stop the server\n")
    print("=" * 50)
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
