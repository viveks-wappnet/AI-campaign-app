import os
import requests
from dotenv import load_dotenv
from typing import Any, Dict, List
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable
from utils.prompt import search_terms_prompt, search_terms_parser, rank_videos_prompt, rank_video_parser
load_dotenv()

# Initialize once
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
groq_llm = ChatGroq(
    model="mistral-saba-24b",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

search_chain: Runnable = search_terms_prompt | groq_llm | search_terms_parser
rank_chain: Runnable = rank_videos_prompt | groq_llm | rank_video_parser

def find_video_url(desc: str) -> str:
    """
    Helper function to find the best video URL for a scene description.
    Returns the video URL or None if no video found.
    """
    # 1) Generate 3 stock-video queries
    terms = search_chain.invoke({"scene_description": desc}).queries

    # 2) Fetch hits for each query
    hits: List[Dict[str, Any]] = []
    for term in terms:
        resp = requests.get(
            "https://pixabay.com/api/videos/",
            params={"key": PIXABAY_API_KEY, "q": term, "per_page": 20}
        )
        resp.raise_for_status()
        hits.extend(resp.json().get("hits", []))

    # Deduplicate
    unique = {v["id"]: v for v in hits}.values()
    hits = list(unique)
    if not hits:
        return None

    # 3) Build the `options` JSON for ranking
    options = []
    for i, v in enumerate(hits[:10]):
        options.append({
            "id"        : i,
            "tags"      : v.get("tags",""),
            "duration"  : v.get("duration",0),
            "resolution": f"{v['videos']['large']['width']}x{v['videos']['large']['height']}",
            "views"     : v.get("views",0)
        })
    video_info = {"options": options}

    # 4) Pick best index via LLM
    best_index = rank_chain.invoke({
        "scene_description": desc,
        "video_info"       : video_info
    }).best_index

    best = hits[min(best_index, len(hits)-1)]

    # 5) Choose highest-res file and return its URL
    files = best["videos"]
    best_file = max(files.values(), key=lambda f: f["width"]*f["height"])
    return best_file["url"]

def generate_video_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: given a script with multiple scenes, finds and appends
    the best matching video URL for each scene.
    
    Input state should contain:
    {
        "script": {
            "scenes": [
                {
                    "scene_id": int,
                    "visual_description": str,
                    ...
                },
                ...
            ]
        }
    }
    
    Returns the script with video_url appended to each scene.
    """
    script = state["script"]
    
    # Process each scene in the script
    for scene in script["scenes"]:
        # Find best video URL for this scene
        video_url = find_video_url(scene["visual_description"])
        if video_url:
            scene["video_url"] = video_url
    
    return {"script": script}
