from pydantic import BaseModel


class CTScan(BaseModel):
    study_id: str
    file_path: str
    modality: str = "CT"
