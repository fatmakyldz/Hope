"""Continual learning metrics."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ContinualMetrics:
    num_tasks: int
    # task_acc_matrix[t][i] = accuracy on task i after training on task t
    task_acc_matrix: list[list[float]] = field(default_factory=list)
    _best_acc: dict[int, float] = field(default_factory=dict)

    def record(self, after_task: int, accs: list[float]) -> None:
        self.task_acc_matrix.append(accs)
        for i, acc in enumerate(accs):
            if acc > self._best_acc.get(i, 0.0):
                self._best_acc[i] = acc

    @property
    def avg_accuracy(self) -> float:
        if not self.task_acc_matrix:
            return 0.0
        return sum(self.task_acc_matrix[-1]) / len(self.task_acc_matrix[-1])

    @property
    def avg_forgetting(self) -> float:
        if len(self.task_acc_matrix) <= 1:
            return 0.0
        final = self.task_acc_matrix[-1]
        fgt = [self._best_acc.get(i, 0.0) - final[i] for i in range(len(final) - 1)]
        return sum(fgt) / len(fgt) if fgt else 0.0

    @property
    def per_task_forgetting(self) -> list[float]:
        final = self.task_acc_matrix[-1] if self.task_acc_matrix else []
        return [self._best_acc.get(i, 0.0) - final[i] for i in range(len(final))]

    def print_matrix(self) -> None:
        print("\nAccuracy matrix (row=after task t, col=task i):")
        header = "       " + "  ".join(f"T{i:02d}" for i in range(len(self.task_acc_matrix[-1])))
        print(header)
        for t, row in enumerate(self.task_acc_matrix):
            vals = "  ".join(f"{v:5.1f}" for v in row)
            print(f"  T{t:02d}: {vals}")

    def summary(self) -> str:
        return (
            f"Avg Accuracy : {self.avg_accuracy:.2f}%\n"
            f"Avg Forgetting: {self.avg_forgetting:.2f}%\n"
            f"Per-task fgt : {[f'{v:.1f}' for v in self.per_task_forgetting]}"
        )

    def save(self, path: str) -> None:
        data = {
            "task_acc_matrix": self.task_acc_matrix,
            "per_task_forgetting": self.per_task_forgetting,
            "avg_accuracy": self.avg_accuracy,
            "avg_forgetting": self.avg_forgetting,
        }
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"Metrics saved -> {path}")
