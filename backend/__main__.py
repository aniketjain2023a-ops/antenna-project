"""Start the API (and the built website, if present):  python -m backend"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000)
