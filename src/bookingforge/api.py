"""Local-only inference API for exported BookingForge research classifiers.

This API intentionally does not train or fit models. At startup, it loads
existing SHA256-verified, pickle-free original NumPy MLP artifacts saved by
bookingforge.model_artifact. Inputs are *synthetic* historical 2015-2016
booking-like records, and outputs are retrospective research scores only.

Default Uvicorn command binds to 127.0.0.1. There is no authentication or
authorization, so NEVER expose this service on a public interface, and do
not use its scores in hotel operational, pricing, staffing, or overbooking
decisions. Historical original booking-time feature snapshots do not exist.
"""
from __future__ import annotations

import calendar
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .arrival_benchmark import TRAIN_FIRST_ARRIVAL, VALID_LAST_ARRIVAL
from .model_artifact import load_portable

RESEARCH_WARNING = (
    "Research-only retrospective arrival-cohort model output. The score is "
    "NOT validated or calibrated for prediction when a reservation is created. "
    "Do not use it for real hotel operations or decisions."
)

Months = Literal[
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
Hotel = Literal["City Hotel", "Resort Hotel"]
RoomType = Literal["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
Variant = Literal["with_year", "without_year"]


class ResearchBooking(BaseModel):
    """Explicit historical feature allowlist; do not accept outcome fields."""

    model_config = ConfigDict(extra="forbid", strict=True)

    hotel: Hotel
    lead_time: int = Field(ge=0, le=10000)
    arrival_date_year: Literal[2015, 2016]
    arrival_date_month: Months
    arrival_date_day_of_month: int = Field(ge=1, le=31)
    stays_in_weekend_nights: int = Field(ge=0, le=1000)
    stays_in_week_nights: int = Field(ge=0, le=1000)
    adults: int = Field(ge=0, le=1000)
    children: int = Field(ge=0, le=1000)
    babies: int = Field(ge=0, le=1000)
    reserved_room_type: RoomType

    @model_validator(mode="after")
    def historical_date_only(self):
        month_num = list(calendar.month_name).index(self.arrival_date_month)
        try:
            day = date(self.arrival_date_year, month_num, self.arrival_date_day_of_month)
        except ValueError as exc:
            raise ValueError("Invalid historical arrival date") from exc
        if not TRAIN_FIRST_ARRIVAL <= day <= VALID_LAST_ARRIVAL:
            raise ValueError(
                "Demo API only accepts historical arrivals from 2015-07-01 "
                "through 2016-12-31"
            )
        return self


class ResearchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    variant: Variant
    booking: ResearchBooking


class ResearchPredictionResponse(BaseModel):
    research_cancellation_score: float
    variant: Variant
    arrival_year_used_by_model: bool
    research_only: Literal[True]
    calibrated_for_booking_time: Literal[False]
    warning: str


def create_app(artifact_dir: str | Path = "models/arrival_v1") -> FastAPI:
    """Return a local research app; load both artifacts once at startup.

    Startup fails closed if weights or metadata are absent, tampered, or
    incompatible. No model is retrained or silently replaced.
    """
    root = Path(artifact_dir)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.research_models = {
            "with_year": load_portable(root, "arrival_mlp_with_year"),
            "without_year": load_portable(root, "arrival_mlp_without_year"),
        }
        try:
            yield
        finally:
            del application.state.research_models

    application = FastAPI(
        title="BookingForge Local Research API",
        description=RESEARCH_WARNING,
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    @application.get("/health", tags=["research"])
    def health():
        return {
            "status": "ready",
            "research_only": True,
            "loaded_variants": sorted(application.state.research_models),
        }

    @application.get("/v1/research/models", tags=["research"])
    def models():
        return {
            "research_only": True,
            "models": {
                variant: {
                    "features_used_for_prediction": list(model.features),
                    "training_rows": model.metadata["training_records"],
                    "dataset_sha256": model.metadata["dataset_sha256"],
                    "arrival_year_used_by_model": model.metadata["include_arrival_year"],
                    "warning": RESEARCH_WARNING,
                }
                for variant, model in application.state.research_models.items()
            },
        }

    @application.post(
        "/v1/research/predict",
        response_model=ResearchPredictionResponse,
        tags=["research"],
    )
    def predict(request: ResearchPredictionRequest):
        saved = application.state.research_models[request.variant]
        data = request.booking.model_dump()
        score = float(saved.predict_proba_positive([data])[0])
        return ResearchPredictionResponse(
            research_cancellation_score=score,
            variant=request.variant,
            arrival_year_used_by_model=(request.variant == "with_year"),
            research_only=True,
            calibrated_for_booking_time=False,
            warning=RESEARCH_WARNING,
        )

    return application


# Uvicorn imports this module without touching model files until startup.
app = create_app()
