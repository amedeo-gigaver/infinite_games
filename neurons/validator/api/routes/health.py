from fastapi import APIRouter

router = APIRouter()


@router.get(
    "",
)
def health():
    return {"status": "OK"}
