from enum import Enum

from models.baseline_qwen_3_model import Qwen3Model
from models.self_rag_model import SelfRAGModel
from models.divide_conquer_model import DivideConquerModel

class Models(Enum):
    BASELINE_QWEN_3 = "baseline_qwen_3"
    SELF_RAG = "self_rag"
    DIVIDE_CONQUER = "divide_conquer"

    # Add more models here

    @staticmethod
    def get_model(model_name: str):
        model = Models(model_name)
        if model == Models.BASELINE_QWEN_3:
            return Qwen3Model
        elif model == Models.SELF_RAG:
            return SelfRAGModel
        elif model == Models.DIVIDE_CONQUER:
            return DivideConquerModel
        else:
            raise ValueError(f"Model `{model_name}` not found.")
