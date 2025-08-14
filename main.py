from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

@app.get("/")
def root():
    return {"status": "ok"}

# make it *obvious* when startup ran
@app.on_event("startup")
async def on_startup():
    print(">>> FASTAPI STARTED <<<")
