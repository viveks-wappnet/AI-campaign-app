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

def generate_video_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: given a scene dict with:
      - scene_id: int
      - visual_description: str
    returns the best Pixabay clip metadata.
    """
    # import ipdb; ipdb.set_trace()
    scene_id = state["scene_id"]
    desc     = state["visual_description"]

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
        return {"scene_id": scene_id, "error": "no_videos_found"}

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

    # 5) Choose highest-res file
    files = best["videos"]
    best_file = max(files.values(), key=lambda f: f["width"]*f["height"])

    # 6) Return
    return {
        "scene_id"   : scene_id,
        "search_query": terms[0] if terms else "",
        "pixabay_id" : best["id"],
        "page_url"   : best["pageURL"],
        "video_url"  : best_file["url"],
        "thumbnail"  : best_file["thumbnail"],
        "duration_s" : best.get("duration"),
        "tags"       : best.get("tags"),
        "resolution" : f"{best_file['width']}x{best_file['height']}",
        "file_size"  : best_file["size"]
    }
