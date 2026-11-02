"""Data evaluation contract; intentionally conservative about availability timing."""
from dataclasses import dataclass
from datetime import date

FORBIDDEN_FEATURES = frozenset({"is_canceled", "reservation_status", "reservation_status_date"})


@dataclass(frozen=True)
class TemporalSplit:
    train_end: date
    validation_end: date
    test_end: date

    def __post_init__(self):
        if not self.train_end < self.validation_end < self.test_end:
            raise ValueError("Split boundaries must be strictly increasing")

    def assign(self, arrival_date: date) -> str:
        if arrival_date <= self.train_end:
            return "train"
        if arrival_date <= self.validation_end:
            return "validation"
        if arrival_date <= self.test_end:
            return "test"
        return "outside"


def validate_features(features: list[str]) -> None:
    leaked = FORBIDDEN_FEATURES.intersection(features)
    if leaked:
        raise ValueError(f"Outcome features prohibited: {sorted(leaked)}")
