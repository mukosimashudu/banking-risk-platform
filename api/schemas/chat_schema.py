from pydantic import BaseModel


class ChatQuestion(BaseModel):
    question: str