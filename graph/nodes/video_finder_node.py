import os
import requests
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable
from utils.prompt import search_terms_prompt, search_terms_parser, rank_videos_prompt, rank_video_parser

load_dotenv()

PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

groq_llm = ChatGroq(
    model="mistral-saba-24b",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

search_chain: Runnable = search_terms_prompt | groq_llm | search_terms_parser
rank_chain:   Runnable = rank_videos_prompt | groq_llm | rank_video_parser


def find_video_url(desc: str) -> str:
    """
    Given a description, returns (best_query, best_video_url) or None.
    """
    # 1) generate queries
    terms = search_chain.invoke({"scene_description": desc}).queries

    # 2) fetch all hits
    hits: List[Dict[str, Any]] = []
    for term in terms:
        resp = requests.get(
            "https://pixabay.com/api/videos/",
            params={"key": PIXABAY_API_KEY, "q": term, "per_page": 10}
        )
        resp.raise_for_status()
        hits.extend(resp.json().get("hits", []))

    # dedupe
    unique = {v["id"]: v for v in hits}.values()
    hits = list(unique)
    if not hits:
        return None

    # 3) rank top 10
    options = []
    for i, v in enumerate(hits[:10]):
        options.append({
            "id": i,
            "tags": v.get("tags", ""),
            "duration": v.get("duration", 0),
            "resolution": f"{v['videos']['large']['width']}x{v['videos']['large']['height']}",
            "views": v.get("views", 0),
        })
    best_index = rank_chain.invoke({
        "scene_description": desc,
        "video_info": {"options": options}
    }).best_index

    best = hits[min(best_index, len(hits) - 1)]
    files = best["videos"]
    best_file = max(files.values(), key=lambda f: f["width"] * f["height"])
    return best_file["url"]


def generate_video_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: takes script with sub_scenes and returns the same structure
    with `search_query` and `video_url` filled in on each sub_scene.
    """
    script = state["script"]

    for scene in script["scenes"]:
        for sub in scene["sub_scenes"]:
            result = find_video_url(sub["visual_description"])
            if result:
                url = result
            else:
                url = None

            # annotate the sub_scene
            sub["video_url"] = url

    return {"script": script}
