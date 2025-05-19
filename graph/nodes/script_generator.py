from typing import Any, Dict
from langchain_groq import ChatGroq
from utils.models import ScriptOutput
from langchain_core.runnables import Runnable
from utils.prompt import script_prompt, script_parser

# Initialize the Groq LLM model
groq_llm = ChatGroq(
    model="mistral-saba-24b",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

script_chain: Runnable = script_prompt | groq_llm | script_parser

def generate_script_node(state: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = state["user_prompt"]
    script_output: ScriptOutput = script_chain.invoke({"user_prompt": user_prompt})
    return {"script": script_output.model_dump()}