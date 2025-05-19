from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from utils.models import ScriptOutput, SearchTermsOutput, RankVideoOutput

# 1) Script generation prompt
script_parser = PydanticOutputParser(pydantic_object=ScriptOutput)
script_prompt = PromptTemplate(
    template="""
You are an expert ad scriptwriter.

Your task is to help a client transform their campaign idea into a **unique and imaginative** 30-second (or whatever length the client wants) video script.

This video should stand out — use bold visuals, a compelling story arc (even if it's just 4-5 scenes for 30 seconds idea but more scenes for bigger idea), and integrate the brand **organically** into the narrative, not as a hard sell.
Think cinematic, emotional, funny, surreal, or metaphorical — whatever brings the brand to life in a **fresh and memorable** way.

For each scene, break down the visual description into sub-scenes that can be used to find specific video clips. Each sub-scene should focus on a distinct visual moment or action that can be found in stock video libraries.

Your output should be in JSON format with 6-7 creative scenes (total duration ≈15 seconds), following this structure:

[
  {{
    "scene_id": 1,
    "duration": "e.g. 5s",
    "dialogue": "Voice-over or character dialogue that can be spoken within the scene's duration. Use tone to match the idea: witty, emotional, bold, mysterious, etc. If necessary include dashes (- or —) for short pauses or ellipses (…) for hesitant tones.",
    "on_screen_text": "Any text appearing on screen: taglines, punchlines, product names, etc.",
    "sub_scenes": [
      {{
        "sub_id": 1,
        "duration": "e.g. 3s",
        "visual_description": "A specific visual moment or action that can be found in stock video. Focus on concrete, searchable elements like locations, actions, objects, or people."
      }},
      {{
        "sub_id": 2,
        "duration": "e.g. 2s",
        "visual_description": "Another distinct visual moment or action for the same scene."
      }}
    ]
  }},
  ...
]

**Important:** 
1. Each scene should have 1-3 sub-scenes that break down the visual narrative into specific, searchable moments
2. Sub-scene descriptions should be concrete and focused on elements that can be found in stock video libraries (locations, actions, objects, people)
3. Avoid abstract or artistic terms in sub-scene descriptions (e.g., "montage", "cinematic", "emotion")
4. The `dialogue` length MUST be 3-4 seconds longer than the scene's `duration` when spoken at a natural pace
5. Each sub-scene should represent a distinct visual moment that can be matched with a stock video clip

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
You need to find ONE perfect video clip that conveys the essence of this scene:

"{scene_description}"

1. If an exact literal match (e.g. a glowing smart water bottle) isn't available on Pixabay, think of **metaphors** or **thematic substitutes** (e.g. water droplets on glass, glowing orb in a dark room, a person taking a refreshing drink).
2. Focus on the **mood**, **key action**, and **distinctive visuals**.
3. Generate exactly 3 highly specific search queries.
4. Return them as a JSON array under the key "queries" (no extra text).

Example output:
{{"queries": ["glowing orb dark room", "water droplets on glass surface", "person drinking water close-up"]}}
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