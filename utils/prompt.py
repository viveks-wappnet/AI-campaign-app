from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from utils.models import ScriptOutput, SearchTermsOutput, RankVideoOutput

# 1) Script generation prompt
script_parser = PydanticOutputParser(pydantic_object=ScriptOutput)
script_prompt = PromptTemplate(
    template="""
You are an expert ad scriptwriter.

Your task is to help a client transform their campaign idea into a **unique and imaginative** 30-second (or whatever length the client wants) video script.

This video should stand out — use bold visuals, a compelling story arc (even if it's just 4-5 scenes), and integrate the brand **organically** into the narrative, not as a hard sell.
Think cinematic, emotional, funny, surreal, or metaphorical — whatever brings the brand to life in a **fresh and memorable** way.

For each scene, in addition to the usual fields, **provide an array of the top most prioritized search‑focused keywords** that best capture the visual concept (short, noun‑based, 1–2 words each) so we can use them directly to query stock video APIs. Avoid including uncommon or abstract terms like 'montage', 'cinematic', 'emotion', or 'aesthetic' — use words likely to appear as tags in stock video libraries like locations, actions, objects, people, or nature terms.

Your output should be in JSON format with 2–4 creative scenes (total duration ≈15 seconds), following this structure:

[
  {{
    "scene": 1,
    "duration": "e.g. 5s",
    "visual_description": "Describe what the viewer sees — set the scene with mood, movement, and atmosphere.",
    "dialogue": "Voice-over or character dialogue. Use tone to match the idea: witty, emotional, bold, mysterious, etc.",
    "on_screen_text": "Any text appearing on screen: taglines, punchlines, product names, etc.",
    "search_query": "concise, noun‑based phrase (2–4 words) ideal for querying stock video libraries"
  }},
  ...
]

**Important:** The `search_query` should be a **single, highly relevant phrase**, composed of terms likely found as tags in stock‑video APIs. Avoid abstract or artistic words (e.g., “montage”, “cinematic”, “emotion”).

{format_instructions}

Client's campaign idea: {user_prompt}
""",
    input_variables=["user_prompt"],
    partial_variables={"format_instructions": script_parser.get_format_instructions()}
)

# ——— VIDEO FINDER ———
# 1) Search terms prompt
search_terms_parser = PydanticOutputParser(pydantic_object=SearchTermsOutput)
search_terms_prompt = PromptTemplate(
    template="""
I need to find ONE perfect video clip that precisely matches this scene description:

"{scene_description}"

Generate 3 highly specific search queries for Pixabay.
Focus on distinctive visual elements, actions, and settings.
List only the queries as a JSON array under the key "queries".

For example:
{{"queries": ["sunset beach", "palm trees", "gentle waves"]}}

""",
    input_variables=["scene_description"],
    partial_variables={"format_instructions": search_terms_parser.get_format_instructions()}
)

# 2) Ranking prompt
rank_video_parser = PydanticOutputParser(pydantic_object=RankVideoOutput)
rank_videos_prompt = PromptTemplate(
    template="""
I need to find THE SINGLE BEST video clip that perfectly matches this scene description:

"{scene_description}"

Here are potential video options from Pixabay (up to 10), formatted as a JSON list under the key "options":

{video_info}

Carefully analyze the tags and attributes. Then return ONLY the JSON output, nothing else.

Output format:
{{"best_index": <zero-based index of best video>}}

DO NOT explain your reasoning.
DO NOT include any text before or after the JSON.
""",
    input_variables=["scene_description", "video_info"],
    partial_variables={"format_instructions": rank_video_parser.get_format_instructions()}
)