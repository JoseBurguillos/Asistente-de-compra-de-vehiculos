from fastapi import FastAPI

app = FastAPI(title="MotorWise API")


@app.get("/api/health")
def health():
    return {"status": "ok"}