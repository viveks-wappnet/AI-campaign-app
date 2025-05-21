from pydantic import BaseModel
from typing import List, Optional

class SubScene(BaseModel):
    sub_id: int
    visual_description: str
    dialogue: str

# Define the Pydantic models for the script output
class Scene(BaseModel):
    scene_id: int
    on_screen_text: str
    sub_scenes: List[SubScene]

class ScriptOutput(BaseModel):
    scenes: List[Scene]

# Define the Pydantic models for the video finder output
class SearchTermsOutput(BaseModel):
    queries: List[str]

class SearchQueryOutput(BaseModel):
    queries: List[str]  # List of search query strings

class RankVideoOutput(BaseModel):
    best_index: int  # zero‑based index of the single best clip