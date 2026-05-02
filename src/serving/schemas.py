from __future__ import annotations

from pydantic import BaseModel, Field


class BikeRecord(BaseModel):
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
    prediction: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class BatchPredictionItem(BaseModel):
    prediction: float
    confidence: float = Field(..., ge=0.0, le=1.0)


class BatchPredictRequest(BaseModel):
    records: list[BikeRecord] = Field(..., min_length=1)


class BatchPredictResponse(BaseModel):
    predictions: list[BatchPredictionItem]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
