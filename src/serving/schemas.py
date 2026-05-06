from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BikeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    season: int = Field(..., ge=1, le=4)
    mnth: int = Field(..., ge=1, le=12)
    hr: int = Field(..., ge=0, le=23)
    holiday: int = Field(..., ge=0, le=1)
    weekday: int = Field(..., ge=0, le=6)
    workingday: int = Field(..., ge=0, le=1)
    weathersit: int = Field(..., ge=1, le=4)
    temp: float = Field(..., ge=0.0, le=1.0)
    atemp: float = Field(..., ge=0.0, le=1.0)
    hum: float = Field(..., ge=0.0, le=1.0)
    windspeed: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class BatchPredictionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: float
    confidence: float = Field(..., ge=0.0, le=1.0)


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[BikeRecord] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions: list[BatchPredictionItem]
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    model_name: str
    model_version: str
