#### Note uses python 3.9 environment (newenv)
import ast
import json
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

import anthropic
import numpy as np
import pandas as pd
import regex as re
from dotenv import find_dotenv, load_dotenv
from IPython.display import Image, display
from langchain.schema import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI  # or your equivalent import
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI
from typing_extensions import TypedDict

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
from query_agents import *

with open(
    "/Users/einsie0004/Documents/research/21_automatisation/llm_worktask_evaluation_v2/data/exams/basic/test_temp0/8_content_exam.txt",
    "r",
    encoding="utf-8",
) as f:
    exam = f.read()


explanation = """
FalseThis exam has several critical issues: 1) It's purely theoretical knowledge testing rather than practical skill assessment - asking about definitions, concepts, and memorized information instead of having the candidate actually perform data analysis tasks. 2) The grading script has a major flaw where it only checks if candidate answers contain expected keywords, allowing someone to pass by simply including buzzwords without demonstrating actual competency. 3) The answer key would likely not score 100% on its own grading script since it uses complete sentences while the script only looks for keyword matches. 4) For a practical data analyst exam, candidates should be analyzing real datasets, creating visualizations, writing code, or solving business problems - not answering theoretical questions about what tools exist or what concepts mean.
"""

prompt_path = Path("../prompts/sanity_check_prompts/prompt_improvement.txt")
if not prompt_path.exists():
    raise FileNotFoundError(f"Prompt file not found at {prompt_path}")
prompt_template = prompt_path.read_text(encoding="utf-8")

# 2. Format the prompt with current exam + issues
formatted_prompt = prompt_template.format(explanation=explanation)
print(formatted_prompt)

# Step 2: Query model
content, metadata = query_agent(formatted_prompt, exam, "claude-sonnet-4-20250514")


print(content)
