from fastapi import APIRouter
from app.services.generation import generate_quiz_universe
from app.schemas.generation import QuizGenerationRequest
import uuid

router = APIRouter(tags=["gen"], prefix="/gen")

@router.post("/quizzes")
async def gen_quizzes(req: QuizGenerationRequest):
    request_id = str(uuid.uuid4())
    output_path = generate_quiz_universe(req, request_id)
    return {"PathToQuizzes": output_path, "RequestID": request_id}


# @router.post("/mcqs")
# async def gen_mcqs(req: QuizGenerationRequest):
#     request_id = str(uuid.uuid4())
#     # TODO: Implement the logic to generate MCQs based on the request
#     # For now, we return a placeholder response
#     return {"Hello": "World!", "RequestID": request_id}