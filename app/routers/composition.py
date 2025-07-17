from fastapi import APIRouter
from app.services.composition import compose_quiz
from app.schemas.composition import QuizCompositionRequest
import uuid

router = APIRouter(tags=["compose"], prefix="/compose")

@router.post("/quiz")
async def gen_quizzes(req: QuizCompositionRequest):
    request_id = str(uuid.uuid4())
    output_path = compose_quiz(req, request_id)
    return {"PathToQuiz": output_path, "RequestID": request_id}
