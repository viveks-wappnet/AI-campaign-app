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

### Structure

Break the story into 5–7 creative **scenes**, each consisting of 1–3 **sub-scenes**.  
Each sub-scene is a concrete **visual moment** with a corresponding **voice-over line** (dialogue). Sub-scenes are the smallest creative units — each will be paired with one stock video and one audio clip.

You do **not** need to specify durations — we will derive them automatically from the spoken dialogue using TTS.

### Output format

Return JSON using this structure(Don't give me the markdown format with the ```JSON```):

{{
  "scenes": [
    {{
      "scene_id": 1,
      "on_screen_text": "Optional: any tagline or product mention shown on screen",
      "sub_scenes": [
        {{
          "sub_id": 1,
          "visual_description": "A specific, searchable visual moment (e.g., 'A woman running through a sunflower field at sunset')",
          "dialogue": "A single, standalone voice-over line that pairs naturally with this moment"
        }},
        {{
          "sub_id": 2,
          "visual_description": "Another distinct, concrete visual moment",
          "dialogue": "Another line of voice-over (distinct from the first)"
        }}
      ]
    }},
    ...
  ]
}}

**Important:**
1. Ensure each `sub_scenes[n].dialogue` contains at least 15-20 words.
2. Add SSML pause tags like <break time=\\"1.5s\\" /> between logical thoughts for natural pacing. For example: "dialogue": "This smart bottle isn't just eco-friendly—it tracks your hydration, syncs to your phone, and glows when it's time to drink. <break time=\\"1.2s\\" /> With a sleek, modern design, it's your new health companion."
3. Do not exceed 3 seconds of pause duration.

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