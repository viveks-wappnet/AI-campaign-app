from pydantic import BaseModel
from typing import List

# Define the Pydantic models for the script output
class Scene(BaseModel):
    scene_id: int
    duration: str
    visual_description: str
    dialogue: str
    on_screen_text: str
    search_query: str

class ScriptOutput(BaseModel):
    scenes: List[Scene]

# Define the Pydantic models for the video finder output
class SearchTermsOutput(BaseModel):
    queries: List[str]

class RankVideoOutput(BaseModel):
    best_index: int  # zero‑based index of the single best clip