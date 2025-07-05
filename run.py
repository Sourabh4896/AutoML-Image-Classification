import subprocess
import sys
import os

def launch_streamlit():
    script_path = os.path.join(os.path.dirname(__file__), "app.py")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])
    except Exception as e:
        print(f"❌ Failed to start Streamlit app: {e}")

if __name__ == "__main__":
    launch_streamlit()
