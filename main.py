# main.py (temporary)
from fastapi import FastAPI
from fastapi.responses import JSONResponse

print(">>> importing main.py")
app = FastAPI()

@app.get("/health")
def health():
    print(">>> /health hit")
    return JSONResponse({"status": "ok"})

@app.get("/")
def root():
    return {"status": "ok"}

@app.on_event("startup")
async def started():
    print(">>> FASTAPI STARTED <<<")
