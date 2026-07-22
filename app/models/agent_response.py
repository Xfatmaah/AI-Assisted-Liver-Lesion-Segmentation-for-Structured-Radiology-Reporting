from pydantic import BaseModel

class AgentResponse(BaseModel):
    success: bool
    report: str