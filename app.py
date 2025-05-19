from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from module.script import generate_ad_script
from utils.db_config import store_script_in_db

# Initialize FastAPI app
app = FastAPI(
    title="Video Ad Script Generator API",
    description="API for generating creative ad scripts using LLM",
    version="0.1.0",
)

# Define request and response models
class ScriptRequest(BaseModel):
    campaign_idea: str = Field(..., description="The campaign idea or concept for the ad")
    
class SceneModel(BaseModel):
    scene: int
    duration: str
    visual_description: str
    dialogue: Optional[str] = None
    on_screen_text: Optional[str] = None
    
class ScriptResponse(BaseModel):
    campaign_idea: str
    script: List[Dict[str, Any]]
    
# Root endpoint
@app.get("/")
async def root():
    """Root endpoint that returns a welcome message."""
    return {"message": "Welcome to the Video Ad Script Generator API"}

# Script generation endpoint
@app.post("/generate-script", response_model=ScriptResponse)
async def create_script(request: ScriptRequest):
    """
    Generate an ad script based on the provided campaign idea.
    
    The script is generated using an LLM and stored in the database.
    """
    try:
        # Generate the script using the LLM
        script = generate_ad_script(request.campaign_idea)
        
        # Store the script in the database
        store_script_in_db(request.campaign_idea, script)
        
        # Return the response
        return {
            "campaign_idea": request.campaign_idea,
            "script": script
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating script: {str(e)}")

# Run the application with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
