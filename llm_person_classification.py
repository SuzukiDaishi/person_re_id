#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Literal

from ollama import Client
from pydantic import BaseModel, Field


# 1. Define Pydantic schema for validation
class PersonAnalysis(BaseModel):
    gender: Literal["male", "female"]
    gender_confidence: float = Field(ge=0, le=100)
    age_category: Literal[
        "Early Childhood (0-4 years)",
        "Middle Childhood (5-14 years)",
        "Adolescence (15-24 years)",
        "Early Adulthood (25-44 years)",
        "Middle Adulthood (45-64 years)",
        "Late Adulthood (65+ years)",
    ]
    age_confidence: float = Field(ge=0, le=100)


def main(image_path: str):
    # 2. Validate image exists
    img = Path(image_path)
    if not img.exists():
        sys.exit(f"Error: file not found: {image_path}")

    # 3. Create remote client
    client = Client(host="http://10.0.1.5:11434")

    # 4. Build JSON schema
    schema = PersonAnalysis.model_json_schema()

    # 5. Call gemma3:4b via client.chat
    response = client.chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "system",
                "content": "Respond only with JSON matching the schema.",
            },
            {
                "role": "user",
                "content": (
                    "Analyze the image for gender ('male' or 'female') and age category "
                    "('Early Childhood (0-4 years)', 'Middle Childhood (5-14 years)', "
                    "'Adolescence (15-24 years)', 'Early Adulthood (25-44 years)', "
                    "'Middle Adulthood (45-64 years)', 'Late Adulthood (65+ years)'), "
                    "and return each with a confidence score (0–100)."
                ),
                "images": [image_path],
            },
        ],
        format=schema,
        options={"temperature": 0},
    )

    # 6. Validate and pretty-print
    result = PersonAnalysis.model_validate_json(response.message.content)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <image_path>")
    main(sys.argv[1])
