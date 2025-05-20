import os
import logging
import requests
from typing import Dict, Any, Optional, List
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable
from utils.prompt import search_terms_prompt, search_terms_parser, rank_videos_prompt, rank_video_parser
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

# Initialize LLM
groq_llm = ChatGroq(
    model="mistral-saba-24b",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize chains
search_chain: Runnable = search_terms_prompt | groq_llm | search_terms_parser
rank_chain: Runnable = rank_videos_prompt | groq_llm | rank_video_parser

def find_video_url(desc: str) -> Optional[str]:
    """
    Given a description, find and return the best matching video URL from Pixabay.
    Returns None if no suitable video is found.
    """
    logger.info(f"Searching for video matching: {desc[:40]}...")
    
    # Generate search queries
    terms = search_chain.invoke({"scene_description": desc}).queries
    logger.debug(f"Generated search terms: {terms}")

    # Fetch video hits
    hits: List[Dict[str, Any]] = []
    for term in terms:
        try:
            resp = requests.get(
                "https://pixabay.com/api/videos/",
                params={"key": PIXABAY_API_KEY, "q": term, "per_page": 5}
            )
            resp.raise_for_status()
            hits.extend(resp.json().get("hits", []))
        except requests.RequestException as e:
            logger.warning(f"Error searching for term '{term}': {str(e)}")
            continue

    # Deduplicate hits
    unique = {v["id"]: v for v in hits}.values()
    hits = list(unique)
    if not hits:
        logger.warning("No video hits found")
        return None

    # Rank top videos
    options = []
    for i, v in enumerate(hits[:10]):
        options.append({
            "id": i,
            "tags": v.get("tags", ""),
            "duration": v.get("duration", 0),
            "resolution": f"{v['videos']['large']['width']}x{v['videos']['large']['height']}",
            "views": v.get("views", 0),
        })
    
    try:
        best_index = rank_chain.invoke({
            "scene_description": desc,
            "video_info": {"options": options}
        }).best_index

        best = hits[min(best_index, len(hits) - 1)]
        files = best["videos"]
        best_file = max(files.values(), key=lambda f: f["width"] * f["height"])
        
        logger.info(f"Found matching video: {best_file['url']}")
        return best_file["url"]
    except Exception as e:
        logger.error(f"Error ranking videos: {str(e)}")
        return None 