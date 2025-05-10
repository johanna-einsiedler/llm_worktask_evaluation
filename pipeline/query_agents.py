from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import anthropic
import google.generativeai as genai
import os
import requests
from google import genai as ggenai


dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# load api keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def take_test(row, system_prompt_template, exam, model):
    """
    Generates a system prompt based on the given occupation and sends it to the specified model along with an exam prompt.

    Parameters:
    - row (dict): A dictionary containing at least an 'occupation' key.
    - system_prompt_template (str): A string template with a placeholder for the occupation.
    - exam (str): The user prompt or exam question to be answered.
    - model (str): The name or identifier of the language model to use.

    Returns:
    - str: The model's response to the prompt, or an empty string if the response is invalid.
    """
    system_prompt = system_prompt_template.format(occupation=row['occupation'])
    response = query_agent(system_prompt, exam, model)
    try:
        return response[0]
    except:
        return ''


def query_agent(system_prompt, user_prompt, model):
    """
    Routes the prompt to the appropriate model-specific query function based on the model identifier.

    Parameters:
    - system_prompt (str): The context or system-level instruction for the model.
    - user_prompt (str): The main user input or question.
    - model (str): A string indicating which model to use (e.g., 'gpt', 'claude', 'gemini').

    Returns:
    - Any: The response from the selected model, or None if the model is not available.
    """
    if 'gemini' in model:
        response = query_gemini(system_prompt, user_prompt, model)
    if 'o3' in model:
        response = query_o3(system_prompt, user_prompt, model)
    if 'gpt' in model:
        response = query_chatgpt(system_prompt, user_prompt, model)
    if 'deepseek' in model:
        response = query_deepseek(system_prompt, user_prompt, model)
    if 'claude' in model:
        response = query_claude(system_prompt, user_prompt, model)
    try:
        return response
    except Exception as e:
        print("Model not currently available")
        return None



def query_gemini(system_prompt, user_prompt, model='gemini-2.0-flash-thinking-exp', temperature=0):
    """
    Sends a system and user prompt to the Gemini model and returns the response.

    Args:
        system_prompt (str): The system-level instruction or context for the model.
        user_prompt (str): The user-level prompt or question.
        model (str): The name of the Gemini model to use (default: "gemini-2.0-flash-thinking-exp").
        temperature (float): Sampling temperature for generation (default: 0 for deterministic output).

    Returns:
        list: A list containing the model's response text and usage metadata, or
        None if an error occurs.
    """
    print("Quering Gemini: ", model)

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model_gen = genai.GenerativeModel(model)
        
        response = model_gen.generate_content(
            contents=[system_prompt, user_prompt],
            generation_config=genai.GenerationConfig(temperature=temperature)
        )
        print(response.usage_metadata)
        return [response.text, response.usage_metadata]

    except Exception as e:
        print(f"Error: {e}")
        return None

def query_deepseek(system_prompt, user_prompt, model="deepseek-chat", temperature=0):
    """
    Sends a system and user prompt to the DeepSeek model and returns the response.

    Args:
        system_prompt (str): The system-level instruction or context for the model.
        user_prompt (str): The user-level prompt or input question.
        model (str): The name of the DeepSeek model to use (default: "deepseek-chat").
        temperature (float): Sampling temperature for response generation (default: 0 for deterministic output).

    Returns:
        list: A list containing the generated response text and usage metadata, 
              or None if an error occurs during the API call.
    """
    print("Quering DeepSeek: ", model)
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature
        )

        return [response.choices[0].message.content, response.usage]
    except Exception as e:
        print(f"Error: {e}")
        return None
    


def query_chatgpt(system_prompt, user_prompt, model="gpt-4o", temperature=0):
    """
    Sends a system and user prompt to the ChatGPT model (OpenAI API) and returns the response.

    Args:
        system_prompt (str): The system-level instruction or context for guiding the model.
        user_prompt (str): The main user input or query.
        model (str): The OpenAI model to use (default: "gpt-4o").
        temperature (float): Sampling temperature for response generation (default: 0 for deterministic output).

    Returns:
        list: A list containing the generated response text and usage metadata, 
                or None if an error occurs during the API call.
    """
    print("Quering ChatGPT: ", model)
    try:

        client = OpenAI(
            api_key=OPENAI_API_KEY
        )

        response = client.chat.completions.create(
            messages=[
                {"role": "developer", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            model=model,
            temperature=temperature

        )
        return [response.choices[0].message.content, response.usage]
    except Exception as e:
        print(f"Error: {e}")
        return None

def query_o3(system_prompt, user_prompt, model="o3-2025-04-16", temperature=1):
    """
    Sends a system and user prompt to the OpenAI O3 model and returns the response.

    Args:
        system_prompt (str): The system-level instruction or context for the model.
        user_prompt (str): The user-level input or query.
        model (str): The specific O3 model to use (default: "o3-2025-04-16").
        temperature (float): Sampling temperature for response generation (default: 1 for more diverse output).

    Returns:
        list: A list containing the generated response text and usage metadata,
              or None if an error occurs during the API call.
    """
    print("Quering ChatGPT: ", model)

    try:

        client = OpenAI(
            api_key=OPENAI_API_KEY
        )

        response = client.chat.completions.create(
            messages=[
                {"role": "developer", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            model=model,
            temperature=temperature

        )
        return [response.choices[0].message.content, response.usage]
    except Exception as e:
        print(f"Error: {e}")
        return None

def query_claude(system_prompt, user_prompt, model="claude-3-7-sonnet-20250219", temperature=0):
    """
    Sends a system and user prompt to the Claude model (Anthropic API) and returns the response.

    Args:
        system_prompt (str): The system-level instruction or context to guide the model's behavior.
        user_prompt (str): The user-level input or question.
        model (str): The Claude model to use (default: "claude-3-7-sonnet-20250219").
        temperature (float): Sampling temperature for generation (default: 0 for deterministic output).

    Returns:
        list: A list containing the generated response text and usage metadata,
              or None if an error occurs during the API call.
    """
    print("Querying Claude: ", model)
    if model== 'claude-3-sonnet-20240229':
        max_tokens = 4096
    else:
        max_tokens = 8192
    try:
        client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=ANTHROPIC_API_KEY
        )

        response = client.messages.create(
            model=model,
            system = system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return [response.content[0].text, response.usage]

    except Exception as e:
        print(f"Error: {e}")
        return None
