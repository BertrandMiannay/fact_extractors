from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ExtractionResult:
    facts: list[dict]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseExtractionApproach(ABC):
    name: str

    @abstractmethod
    def extract(self, file_path: str) -> ExtractionResult:
        pass
