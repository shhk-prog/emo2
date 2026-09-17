import json
from pydantic import BaseModel, Field, ValidationError

class AffectiveState(BaseModel):
    """Schema for parsing the self-reported affective state."""
    valence: int = Field(..., ge=1, le=9, description="Valence score (1-9)")
    arousal: int = Field(..., ge=1, le=9, description="Arousal score (1-9)")

    model_config = {
        "extra": "forbid"
    }

def parse_affective_state(response_text: str) -> dict:
    """
    Parses the response text into a valid AffectiveState dictionary.
    Returns the parsed dict or raises ValueError if invalid.
    """
    try:
        data = json.loads(response_text)
        state = AffectiveState(**data)
        return state.model_dump()
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {e}")
    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {e}")
