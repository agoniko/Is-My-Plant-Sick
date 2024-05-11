from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys

app = FastAPI()

# Enable CORS for all origins, allow all methods, allow all headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PathModel(BaseModel):
    path:str


@app.post("/path/")
async def create_item(path: PathModel):
    try:
        # Validate input
        if not path.path:
            raise HTTPException(status_code=400, detail="path cannot be empty")

        # Append to the file (use 'a' mode)
        with open("path.txt", "a") as f:
            f.write(path.path + "\n")

        return {"message": "Path added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
