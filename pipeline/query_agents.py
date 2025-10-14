import os

import anthropic
from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
import requests
from together import Together

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# load api keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")


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
    system_prompt = system_prompt_template.format(occupation=row["occupation"])
    response = query_agent(system_prompt, exam, model)
    try:
        return response[0]
    except:
        return ""


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
    if "gemini" in model:
        response = query_gemini(system_prompt, user_prompt, model)
    if "o3" in model:
        response = query_o3(system_prompt, user_prompt, model)
    if "gpt" in model:
        response = query_chatgpt(system_prompt, user_prompt, model)
    if "deepseek" in model:
        response = query_deepseek(system_prompt, user_prompt, model)
    if "claude" in model:
        response = query_claude(system_prompt, user_prompt, model)
    if "llama" in model:
        response = query_open_source(system_prompt, user_prompt, model)
    if "perplexity" in model:
        response = query_open_source(system_prompt, user_prompt, model)
    try:
        return response
    except Exception:
        print("Model not currently available")
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
            temperature=temperature,
        )

        return [response.choices[0].message.content, response.usage]
    except Exception as e:
        print(f"Error: {e}")
        return None


def query_gemini(system_prompt, user_prompt, model="gemini-2.0-flash-thinking-exp", temperature=0):
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

    # HARM_CATEGORY_HATE_SPEECH, HARM_CATEGORY_SEXUALLY_EXPLICIT, HARM_CATEGORY_DANGEROUS_CONTENT, HARM_CATEGORY_HARASSMENT, HARM_CATEGORY_CIVIC_INTEGRITY
    try:
        # genai.configure(api_key=GOOGLE_API_KEY)

        client = genai.Client()

        response = client.models.generate_content(
            # model_gen = genai.GenerativeModel(model)
            # response = model_gen.generate_content(
            model=model,
            contents=[system_prompt, user_prompt],
            config=types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ],
                temperature=temperature,
            ),
            # generation_config=genai.GenerationConfig(temperature=temperature),
        )
        # print("Gemini response: ", response)
        return [response.text, response.usage_metadata]

    except Exception as e:
        print(f"Error: {e}")
        return None


def query_open_source(system_prompt, user_prompt, model="deepseek-ai/DeepSeek-V3", temperature=0):
    """
    Sends a system and user prompt to Together API  model and returns the response.

    Args:
        system_prompt (str): The system-level instruction or context for the model.
        user_prompt (str): The user-level prompt or input question.
        model (str): The name of the DeepSeek model to use (default: "deepseek-chat").
        temperature (float): Sampling temperature for response generation (default: 0 for deterministic output).

    Returns:
        list: A list containing the generated response text and usage metadata,
              or None if an error occurs during the API call.
    """
    print("Quering an open source model via Together AI ", model)
    try:
        client = Together()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
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
        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            messages=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            # temperature=temperature,
            # max_tokens=4000,  # Adjust max tokens as needed
        )
        # print(response)
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
        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            messages=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=temperature,
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
    if model == "claude-3-sonnet-20240229":
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
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return [response.content[0].text, response.usage]

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    system_prompt = "You are an expert in budget planning and administration. Complete the following exam task."
    user_prompt = """
    # Basic Practical Exam: Budget Planning and Administration for Social and Community Service Managers

    ## Overview
    You are a newly hired Social and Community Service Manager at Community Wellness Initiative (CWI), a non-profit organization providing mental health services, youth development programs, and senior support services. Your task is to plan and administer the budget for the upcoming fiscal year for the Youth Development Program.

    ## Time Limit
    90 minutes

    ## Materials Provided
    1. **Previous Year Budget Spreadsheet** (youth_program_previous_budget.xlsx) containing:
    - Line item expenses categorized by department
    - Funding sources and amounts
    - Notes on spending restrictions

    2. **Current Year Funding Document** (current_funding.xlsx) containing:
    - Confirmed funding amounts from all sources
    - Funding restrictions and requirements
    - Reporting deadlines

    3. **Program Requirements Document** (program_requirements.txt) containing:
    - Minimum staffing requirements
    - Required program components
    - Compliance guidelines

    ## Tasks

    ### Task 1: Budget Analysis and Planning (30 points)
    1. Review the previous year's budget spreadsheet and identify any calculation errors or discrepancies
    2. Create a new budget for the upcoming fiscal year using the current funding commitments
    3. Allocate funds across the following categories using the provided template:
    - Staffing costs (salaries and benefits)
    - Program supplies and materials
    - Facility costs
    - Administrative overhead
    - Transportation
    - Technology needs

    ### Task 2: Budget Adjustment (30 points)
    After creating your initial budget, you receive notification that the County Youth Services Grant will be reduced by 15% ($12,000).
    1. Adjust your budget to accommodate this reduction
    2. Ensure all minimum program requirements are still met
    3. Record the specific dollar amount reduced from each budget category

    ### Task 3: Budget Administration (40 points)
    1. Create a quarterly breakdown of your final budget using the provided template
    2. Calculate the percentage of total budget allocated to each expense category
    3. Determine the cost per participant based on serving 120 youth annually
    4. Complete the budget summary statements by filling in the correct numerical values

    ## Submission Format
    Submit your answers in a file named "test_submission.json" with the following structure:

    ```json
    {
    "task1": {
        "discrepancies_identified": ["List specific calculation errors found with dollar amounts"],
        "new_budget": {
        "staffing": 0,
        "program_supplies": 0,
        "facility_costs": 0,
        "administrative": 0,
        "transportation": 0,
        "technology": 0,
        "total_budget": 0
        }
    },
    "task2": {
        "adjusted_budget": {
        "staffing": 0,
        "program_supplies": 0,
        "facility_costs": 0,
        "administrative": 0,
        "transportation": 0,
        "technology": 0,
        "total_budget": 0
        },
        "reduction_by_category": {
        "staffing": 0,
        "program_supplies": 0,
        "facility_costs": 0,
        "administrative": 0,
        "transportation": 0,
        "technology": 0,
        "total_reduction": 0
        }
    },
    "task3": {
        "quarterly_breakdown": {
        "q1": {
            "staffing": 0,
            "program_supplies": 0,
            "facility_costs": 0,
            "administrative": 0,
            "transportation": 0,
            "technology": 0,
            "total_q1": 0
        },
        "q2": {
            "staffing": 0,
            "program_supplies": 0,
            "facility_costs": 0,
            "administrative": 0,
            "transportation": 0,
            "technology": 0,
            "total_q2": 0
        },
        "q3": {
            "staffing": 0,
            "program_supplies": 0,
            "facility_costs": 0,
            "administrative": 0,
            "transportation": 0,
            "technology": 0,
            "total_q3": 0
        },
        "q4": {
            "staffing": 0,
            "program_supplies": 0,
            "facility_costs": 0,
            "administrative": 0,
            "transportation": 0,
            "technology": 0,
            "total_q4": 0
        }
        },
        "budget_metrics": {
        "staffing_percentage": 0,
        "program_supplies_percentage": 0,
        "facility_costs_percentage": 0,
        "administrative_percentage": 0,
        "transportation_percentage": 0,
        "technology_percentage": 0,
        "cost_per_participant": 0
        }
    }
    }
    ```
    ## File: youth_program_previous_budget.xlsx

    ### Sheet 1: Budget Summary
    | Category | Amount | % of Total |
    |----------|--------|------------|
    | Staffing | $85,000 | 53.1% |
    | Program Supplies | $25,000 | 15.6% |
    | Facility Costs | $20,000 | 12.5% |
    | Administrative | $15,000 | 9.4% |
    | Transportation | $10,000 | 6.3% |
    | Technology | $5,000 | 3.1% |
    | **TOTAL** | **$160,000** | **100%** |

    ### Sheet 2: Staffing Detail
    | Position | FTE | Annual Salary | Benefits (25%) | Total Cost |
    |----------|-----|---------------|----------------|------------|
    | Program Director | 0.5 | $50,000 | $12,500 | $25,000 |
    | Program Coordinator | 1.0 | $40,000 | $10,000 | $40,000 |
    | Youth Counselor | 0.5 | $36,000 | $9,000 | $18,000 |
    | Administrative Assistant | 0.25 | $32,000 | $8,000 | $8,000 |
    | **TOTAL** | **2.25** | | | **$91,000** |

    ### Sheet 3: Funding Sources
    | Source | Amount | Restrictions |
    |--------|--------|-------------|
    | County Youth Services Grant | $80,000 | Must allocate at least 60% to direct program costs |
    | Community Foundation | $40,000 | No restrictions |
    | Corporate Sponsor (TechCorp) | $25,000 | $5,000 must be used for technology |
    | Program Fees | $15,000 | No restrictions |
    | **TOTAL** | **$160,000** | |

    ### Sheet 4: Quarterly Expenditures
    | Category | Q1 | Q2 | Q3 | Q4 | Total |
    |----------|----|----|----|----|-------|
    | Staffing | $21,250 | $21,250 | $21,250 | $21,250 | $85,000 |
    | Program Supplies | $5,000 | $7,500 | $7,500 | $5,000 | $25,000 |
    | Facility Costs | $5,000 | $5,000 | $5,000 | $5,000 | $20,000 |
    | Administrative | $3,750 | $3,750 | $3,750 | $3,750 | $15,000 |
    | Transportation | $2,000 | $3,000 | $3,000 | $2,000 | $10,000 |
    | Technology | $2,000 | $1,000 | $1,000 | $1,000 | $5,000 |
    | **TOTAL** | **$39,000** | **$41,500** | **$41,500** | **$38,000** | **$160,000** |

    ## File: current_funding.xlsx

    ### Sheet 1: Confirmed Funding
    | Source | Previous Year | Current Year | Change | Restrictions |
    |--------|---------------|--------------|--------|-------------|
    | County Youth Services Grant | $80,000 | $80,000 | $0 | Must allocate at least 60% to direct program costs (staffing, program supplies, transportation) |
    | Community Foundation | $40,000 | $45,000 | +$5,000 | No restrictions |
    | Corporate Sponsor (TechCorp) | $25,000 | $30,000 | +$5,000 | $10,000 must be used for technology |
    | Program Fees | $15,000 | $15,000 | $0 | No restrictions |
    | **TOTAL** | **$160,000** | **$170,000** | **+$10,000** | |

    ### Sheet 2: Reporting Requirements
    | Funder | Report Due Dates | Required Information |
    |--------|------------------|----------------------|
    | County Youth Services Grant | Quarterly (15 days after quarter end) | Detailed expenditures by category, number of youth served |
    | Community Foundation | Mid-year and Year-end | Budget vs. actual, program outcomes |
    | Corporate Sponsor (TechCorp) | Year-end | Technology purchases and impact |
    | Program Fees | N/A | N/A |

    ## File: program_requirements.txt

    # Youth Development Program Requirements

    ## Minimum Staffing Requirements
    - Program must maintain at least 2.0 FTE staff positions
    - Program Director position must be at least 0.5 FTE
    - At least one full-time Program Coordinator is required
    - Youth Counselor position must be at least 0.5 FTE

    ## Required Program Components
    1. After-school activities (minimum 3 days per week)
    2. Weekend workshops (minimum 2 per month)
    3. Individual counseling sessions (as needed)
    4. Parent engagement activities (minimum quarterly)
    5. Community service projects (minimum 2 per year)

    ## Budget Guidelines
    - Administrative costs should not exceed 10% of total budget
    - At least 50% of budget should be allocated to staffing
    - Technology expenses should be 5-10% of total budget
    - Transportation must be adequately funded to ensure program accessibility
    - Facility costs should include utilities, maintenance, and security

    ## Compliance Requirements
    - All staff must complete background checks
    - Staff-to-youth ratio must not exceed 1:15
    - All activities must be documented with attendance records
    - Quarterly program reports must be submitted to funders
    - Annual program evaluation must be conducted

    ## Program Capacity
    - Program must serve a minimum of 120 youth annually
    - Maximum group size for any activity is 30 youth
    # Answer Format Requirements

    ## JSON Structure
    Submit your answers in a file named "test_submission.json" with the following structure and formatting requirements:

    ```json
    {
    "task1": {
        "discrepancies_identified": ["List each discrepancy as a string with specific dollar amounts"],
        "new_budget": {
        "staffing": 0,
        "program_supplies": 0,
        "facility_costs": 0,
        "administrative": 0,
        "transportation": 0,
        "technology": 0,
        "total_budget": 0
        }
    },
    "task2": {
        "adjusted_budget": {
        "staffing": 0,
        "program_supplies": 0,
        "facility_costs": 0,
        "administrative": 0,
        "transportation": 0,
        "technology": 0,
        "total_budget": 0
        },
        "reduction_by_category": {
        "staffing": 0,
        "program_supplies": 0,
        "facility_costs": 0,
        "administrative": 0,
        "transportation": 0,
        "technology": 0,
        "total_reduction": 0
        }
    },
    "task3": {
        "quarterly_breakdown": {
        "q1": {
            "staffing": 0,
            "program_supplies": 0,
            "facility_costs": 0,
            "administrative": 0,
            "transportation": 0,
            "technology": 0,
            "total_q1": 0
        },
        "q2": {
            "staffing": 0,
            "program_supplies": 0,
            "facility_costs": 0,
            "administrative": 0,
            "transportation": 0,
            "technology": 0,
            "total_q2": 0
        },
        "q3": {
            "staffing": 0,
            "program_supplies": 0,
            "facility_costs": 0,
            "administrative": 0,
            "transportation": 0,
            "technology": 0,
            "total_q3": 0
        },
        "q4": {
            "staffing": 0,
            "program_supplies": 0,
            "facility_costs": 0,
            "administrative": 0,
            "transportation": 0,
            "technology": 0,
            "total_q4": 0
        }
        },
        "budget_metrics": {
        "staffing_percentage": 0,
        "program_supplies_percentage": 0,
        "facility_costs_percentage": 0,
        "administrative_percentage": 0,
        "transportation_percentage": 0,
        "technology_percentage": 0,
        "cost_per_participant": 0
        }
    }
    }
    ```

    ## Format Requirements

    ### Task 1: Budget Analysis and Planning
    - **discrepancies_identified**: List each discrepancy as a separate string in the array, including specific dollar amounts (e.g., "Staffing budget shows $X in summary but $Y in detail sheet")
    - **new_budget**: All monetary values should be integers with no decimal places, no dollar signs, and no commas (e.g., 85000 not $85,000.00)

    ### Task 2: Budget Adjustment
    - **adjusted_budget**: All monetary values should be integers with no decimal places, no dollar signs, and no commas
    - **reduction_by_category**: Enter the exact dollar amount reduced from each category (positive integers only)
    - **total_reduction**: Must equal exactly $12,000

    ### Task 3: Budget Administration
    - **quarterly_breakdown**: All monetary values should be integers with no decimal places, no dollar signs, and no commas
    - **budget_metrics**:
    - All percentage values should be numbers with one decimal place (e.g., 50.5 not 50.5%)
    - **cost_per_participant**: Should be a number with two decimal places (e.g., 1316.67)

    ## Validation Requirements
    - The sum of all category values must equal the total_budget value
    - The sum of all quarterly totals must equal the total annual budget
    - The sum of all percentage values in budget_metrics must equal 100.0
    - All calculations must be mathematically accurate

    No supplementary files are required beyond the test_submission.json file.
    """
    model = "gpt-3.5-turbo-0125"  # Example model name

    response = query_agent(system_prompt, user_prompt, model)
    print("Response:", response)
