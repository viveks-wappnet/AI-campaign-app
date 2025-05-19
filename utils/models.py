from pydantic import BaseModel
from typing import List

# Define the Pydantic models for the script output
class Scene(BaseModel):
    scene: int
    duration: str
    visual_description: str
    dialogue: str
    on_screen_text: str
    search_query: str

class ScriptOutput(BaseModel):
    scenes: List[Scene]

