from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from api.base import APIResponse


@dataclass
class ApproachResponse:
    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_api_response(cls, response: APIResponse) -> "ApproachResponse":
        return cls(
            answer=response.content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )


class BaseApproach(ABC):
    name: str

    @abstractmethod
    def ask(self, question: str) -> ApproachResponse:
        pass
