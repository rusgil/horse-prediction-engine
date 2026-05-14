import uvicorn

if __name__ == "__main__":
    uvicorn.run("horse_engine.api.main:app", host="0.0.0.0", port=8001, reload=True)
