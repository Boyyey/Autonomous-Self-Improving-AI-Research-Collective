# model_zoo_server.py
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import json

app = FastAPI()
templates = Jinja2Templates(directory="model_zoo/templates")
app.mount("/static", StaticFiles(directory="model_zoo/static"), name="static")

@app.get("/")
async def model_zoo(request: Request):
    models = []
    for f in os.listdir("checkpoints"):
        if f.endswith(".pth"):
            gen = f.split("_")[1].split(".")[0]
            meta = {
                "name": f"Agent-Gen{gen}",
                "generation": gen,
                "size_kb": os.path.getsize(f"checkpoints/{f}") // 1024,
                "download_url": f"/static/{f}"
            }
            models.append(meta)
    return templates.TemplateResponse("zoo.html", {"request": request, "models": models})