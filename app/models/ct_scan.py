from pydantic import BaseModel


class CTScan(BaseModel):
    """A single CT scan case to be processed by the pipeline.

    `category` is optional dataset metadata (e.g. a source folder name such
    as "liver_lesion" or "normal" in the original research dataset layout).
    It is carried through purely for evaluation, dataset organization, and
    reporting purposes. It must NEVER be used to decide what the inference
    pipeline is capable of processing or which findings it reports — the
    pipeline always derives findings from the actual segmentation output,
    not from this label.
    """

    study_id: str
    file_path: str
    modality: str = "CT"
    category: str | None = None
