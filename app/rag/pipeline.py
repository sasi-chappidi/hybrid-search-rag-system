from app.rag.answerer_local import answer_locally

def answer_with_rag(question: str, retrieved: list[dict]) -> str:
    return answer_locally(question, retrieved)