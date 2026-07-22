from pydantic import BaseModel

class ToolCall(BaseModel):
    tool_name: str
    input_path: str 