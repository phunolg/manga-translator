from fastapi import FastAPI
import uvicorn
from routes.init_manga_router import init_manga_router
from routes.main_routes import main_routes

app = FastAPI()


# Đăng ký router
app.include_router(init_manga_router)
app.include_router(main_routes)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8008, reload=True)
