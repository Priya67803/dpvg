"""Start the backend + dashboard:  python run.py  ->  http://127.0.0.1:8000"""
import os, uvicorn

if __name__ == "__main__":
    os.environ.setdefault("DPVG_BACKEND", "ollama")
    uvicorn.run("app.main:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")), reload=False)
