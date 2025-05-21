# graph/nodes/video_finder_node.py

import os
import json
import time
import logging
import requests
from typing import Any, Dict, List, Optional
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable
from langchain import PromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from utils.prompt import rank_videos_prompt, rank_video_parser
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SHUTTERSTOCK_TOKEN = os.getenv("SHUTTERSTOCK_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {SHUTTERSTOCK_TOKEN}",
    "Content-Type": "application/x-www-form-urlencoded"
}

# ——— LLM & Chains ———
groq_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.8)

# 1) initial single-query prompt
class SearchQueryOutput(BaseModel):
    query: str = Field(..., description="A single, concise 3–5 word query")

search_query_parser = PydanticOutputParser(pydantic_object=SearchQueryOutput)
search_query_prompt = PromptTemplate(
    input_variables=["scene_description"],
    partial_variables={"format_instructions": search_query_parser.get_format_instructions()},
    template="""
Generate a single, concise (3–5 word) Shutterstock search query that best matches this scene:

"{scene_description}"

Focus on the most distinctive visual element and one contextual cue.

IMPORTANT: Return a SINGLE string query, NOT a list or array. The output should be a simple string like "woman waking up" or "water bottle glow".

Example valid outputs:
{{
  "query": "woman waking up"
}}
or
{{
  "query": "water bottle on a table"
}}

{format_instructions}
"""
)
search_chain: Runnable = search_query_prompt | groq_llm | search_query_parser

# 2) ranking chain (unchanged)
rank_chain: Runnable = rank_videos_prompt | groq_llm | rank_video_parser

# 3) refine prompt with memory
class RefinedQueryOutput(BaseModel):
    query: str = Field(..., description="A single improved search query")

refine_query_parser = PydanticOutputParser(pydantic_object=RefinedQueryOutput)
refine_query_prompt = PromptTemplate(
    input_variables=["scene_description", "history"],
    partial_variables={"format_instructions": refine_query_parser.get_format_instructions()},
    template="""
The following search queries have already been tried (and failed) for this scene:
{history}

Scene description:
"{scene_description}"

You must propose a completely new and most simplest version 3–5 word query that is NOT in the history above and is 100% guaranteed to return results on a stock video site. **Don't just add "too", "at" minor changes; completely change it to the most simplest searchable query!**

Return only valid JSON (no extra text):

{{
  "query": "<new query>"
}}

Example of refinement:
{{
    "Failed query": "smart water bottle notification" 
}}
 and it's refined version:
{{
    "query": "water bottle"
}} 
or
{{
    "query": "notification"
}}


{format_instructions}
"""
)
refine_chain: Runnable = refine_query_prompt | groq_llm | refine_query_parser

# ——— Helper functions ———

def _shutterstock_search(query: str, per_page: int = 10) -> List[Dict[str, Any]]:
    resp = requests.get(
        "https://api.shutterstock.com/v2/videos/search",
        params={"query": query, "per_page": per_page, "view": "full"},
        headers=HEADERS
    )
    resp.raise_for_status()
    return resp.json().get("data", [])

def _rank_and_pick(candidates: List[Dict[str, Any]], desc: str) -> Optional[str]:
    if not candidates:
        return None

    options = []
    for idx, item in enumerate(candidates[:10]):
        options.append({
            "id": idx,
            "description": item["description"],
            "keywords": item.get("keywords", []),
            "categories": [c["name"] for c in item.get("categories", [])],
            "duration": item["duration"],
        })

    best_index = rank_chain.invoke({
        "scene_description": desc,
        "video_info": {"options": options}
    }).best_index

    chosen = candidates[min(best_index, len(candidates) - 1)]
    return chosen["assets"]["preview_mp4"]["url"]

# ——— Core recursive search ———

def find_video_url(
    desc: str,
    max_attempts: int = 10,
    timeout_seconds: int = 60
) -> Optional[str]:
    start = time.time()
    seen: List[str] = []

    # 1) initial query
    initial = search_chain.invoke({"scene_description": desc}).query
    query = initial.strip()
    logger.info(f"Initial query: {query!r}")

    attempts = 0
    while True:
        attempts += 1
        elapsed = time.time() - start
        if elapsed > timeout_seconds:
            logger.error(f"Timeout after {elapsed:.1f}s searching for '{desc}'")
            break
        if attempts > max_attempts:
            logger.error(f"Max attempts {max_attempts} reached for '{desc}'")
            break

        if query in seen:
            logger.warning(f"Query already tried: {query!r}; stopping early")
            break

        seen.append(query)
        logger.info(f"[Attempt {attempts}] Searching Shutterstock for: {query!r}")
        results = _shutterstock_search(query)
        url = _rank_and_pick(results, desc)
        if url:
            logger.info(f"Found video for '{desc}' with query '{query}': {url}")
            return url

        # refine
        history_json = json.dumps(seen, ensure_ascii=False)
        refined = refine_chain.invoke({
            "scene_description": desc,
            "history": history_json
        }).query.strip()
        logger.info(f"Refined query: {refined!r}")
        print(f"Failed query: {query}")
        query = refined

    logger.warning(f"No video found for scene: {desc!r}")
    return None

def generate_video_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # import ipdb; ipdb.set_trace()
    for scene in state["script"]["scenes"]:
        for sub in scene["sub_scenes"]:
            sub["video_url"] = find_video_url(sub["visual_description"])
        print(f"Sub: {sub}")
    return {"script": state["script"]}