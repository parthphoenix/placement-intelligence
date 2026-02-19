"""
Pydantic schemas for the FastAPI prediction endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional


class StudentInput(BaseModel):
    """Input schema matching the feature set used by the models."""
    Gender: str = Field(..., example="Male", description="Male or Female")
    Age: int = Field(..., ge=16, le=50, example=22)
    Degree: str = Field(..., example="Engineering",
                        description="One of: Engineering, Computer Science, Business, Arts, Data Science")
    CGPA: float = Field(..., ge=0.0, le=4.0, example=3.5)
    Internships_Count: int = Field(..., ge=0, le=10, example=2)
    Projects_Count: int = Field(..., ge=0, le=20, example=5)
    Certifications_Count: int = Field(..., ge=0, le=10, example=3)
    Technical_Skills_Score_100: int = Field(..., ge=0, le=100, example=80)
    Communication_Skills_Score_100: int = Field(..., ge=0, le=100, example=70)
    Aptitude_Test_Score_100: int = Field(..., ge=0, le=100, example=75)

    class Config:
        json_schema_extra = {
            "example": {
                "Gender": "Male",
                "Age": 22,
                "Degree": "Engineering",
                "CGPA": 3.5,
                "Internships_Count": 2,
                "Projects_Count": 5,
                "Certifications_Count": 3,
                "Technical_Skills_Score_100": 80,
                "Communication_Skills_Score_100": 70,
                "Aptitude_Test_Score_100": 75,
            }
        }


class PlacementResponse(BaseModel):
    placement_prediction: str = Field(..., description="'Placed' or 'Not Placed'")
    placement_probability: float = Field(..., description="Probability of placement (0-1)")


class SalaryResponse(BaseModel):
    predicted_salary: float = Field(..., description="Predicted salary in USD")
