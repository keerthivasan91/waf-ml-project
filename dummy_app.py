from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Protected backend app is running"}

@app.get("/hello")
def hello():
    return {"message": "Hello from protected app"}

@app.get("/search")
def search(q: str = ""):
    return {"query": q}

@app.get("/health")
def health():
    return {"status": "healthy"}