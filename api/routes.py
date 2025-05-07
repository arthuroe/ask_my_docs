from fastapi import APIRouter

router = APIRouter(tags=["query"])

@router.get("/")
async def index():
    return {"message": "Welcome to ask my doc"}


@router.get("/health")
async def health_check():
    return {"status": "healthy"}