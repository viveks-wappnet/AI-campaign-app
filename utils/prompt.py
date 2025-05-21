from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from utils.models import ScriptOutput, SearchTermsOutput, RankVideoOutput, SearchQueryOutput

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
1. Don’t make the `visual_description` too cinematic. If it sounds overly fictional, it will be much harder to find a matching clip in a stock‑video library. Keep it simple. Instead make the dialogues more dramatic and less boring.
2. Ensure each `sub_scenes[n].dialogue` contains at least 15-20 words.
3. For pauses in dialogue, use the following format: "First part of dialogue [PAUSE:1.2s] Second part of dialogue"
4. Do not exceed 3 seconds of pause duration.
5. DO NOT use SSML tags directly in the dialogue. Instead, use the [PAUSE:X.Xs] format.

{format_instructions}

Client's campaign idea: {user_prompt}
""",
    input_variables=["user_prompt"],
    partial_variables={"format_instructions": script_parser.get_format_instructions()}
)

# ——— VIDEO FINDER ———
# 1) Search terms prompt
search_terms_parser = PydanticOutputParser(pydantic_object=SearchQueryOutput)
search_terms_prompt  = PromptTemplate(
    template="""
Generate a single, concise (3–5 word) Shutterstock search query that best matches this scene:

"{scene_description}"

Focus on the most distinctive visual element and one contextual cue.

{format_instructions}
""",
    input_variables=["scene_description"],
    partial_variables={"format_instructions": search_terms_parser.get_format_instructions()}
)

# 2) Ranking prompt
rank_video_parser = PydanticOutputParser(pydantic_object=RankVideoOutput)
rank_videos_prompt = PromptTemplate(
    template="""
You are given a scene description and a list of candidate video options in JSON format.

Your task is to SELECT THE SINGLE BEST video that matches the scene description based on visual match, quality, and relevance.

DO NOT explain or include the full options.

RETURN ONLY a single JSON object in this exact format:
{{
  "best_index": <number>
}}

For example, if the best match is at index 4:
{{
  "best_index": 4
}}

INPUT:
Scene description: {scene_description}

Video options (JSON under the key "options"): {video_info}

Constraints:
- The response MUST be valid JSON
- DO NOT include markdown formatting
- DO NOT include any explanation or extra text
""",
    input_variables=["scene_description", "video_info"],
    partial_variables={
        "format_instructions": rank_video_parser.get_format_instructions()
    }
)