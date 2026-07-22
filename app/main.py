from fastapi import FastAPI
from pydantic import BaseModel

class Message(BaseModel):
    text: str

app = FastAPI()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/echo")
async def echo():
    return {"message": "Echo endpoint"}

