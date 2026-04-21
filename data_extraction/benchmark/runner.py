import json
from dataclasses import dataclass, field
from pathlib import Path
from data_extraction.approaches.base import BaseApproach
from data_extraction.benchmark.judge import Judge


@dataclass
class QuestionResult:
    short_label: str
    question: str
    expected_answer: str
    actual_answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    judge_score: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ApproachBenchmark:
    approach_name: str
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def avg_total_tokens(self) -> float:
        return sum(r.total_tokens for r in self.results) / len(self.results) if self.results else 0

    @property
    def avg_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.results) / len(self.results) if self.results else 0

    @property
    def avg_judge_score(self) -> float:
        scored = [r for r in self.results if r.judge_score > 0]
        return sum(r.judge_score for r in scored) / len(scored) if scored else 0


class BenchmarkRunner:
    def __init__(self, questions_path: str | None = None):
        path = questions_path or str(Path(__file__).parent / "questions.json")
        with open(path, encoding="utf-8") as f:
            self.questions: list[dict] = json.load(f)

    def run(
        self,
        approach: BaseApproach,
        judge: Judge | None = None,
        progress_callback=None,
    ) -> ApproachBenchmark:
        benchmark = ApproachBenchmark(approach_name=approach.name)
        total = len(self.questions)

        for i, q in enumerate(self.questions):
            response = approach.ask(q["question"])
            result = QuestionResult(
                short_label=q.get("short_label", q["question"][:40]),
                question=q["question"],
                expected_answer=q["expected_answer"],
                actual_answer=response.answer,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
                metadata=response.metadata,
            )
            if judge:
                result.judge_score = judge.score(
                    q["question"], q["expected_answer"], response.answer
                )
            benchmark.results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return benchmark
