from pydantic import BaseModel, ConfigDict, Field


class HeartFeatures(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., gt=0, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., gt=0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., gt=0, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, description="ST depression by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST")
    ca: int = Field(..., ge=0, le=4, description="Major vessels colored (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
                "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
                "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3,
            }
        }
    )


class PredictionResponse(BaseModel):
    heart_disease: bool
    probability: float


class InfoResponse(BaseModel):
    model_type: str
    features: list[str]
    test_accuracy: float | None = None
