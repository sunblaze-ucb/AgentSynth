import argparse
import requests
import json
import time
import re
import random
from tqdm import tqdm
import os
import base64
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from PIL import Image
import sys
import io
import signal
from miscellaneous_funcs import *
from insta import InstaEnv, NULL_ACTION, BrowserStatus
from insta.configs.agent_config import BrowserAction
from insta.configs.browser_config import FunctionCall
from insta.pipeline import generate_trajectory
from insta.configs import get_browser_config, get_agent_config, get_judge_config

# Add a signal handler to gracefully shut down
def handle_sigterm(signum, frame):
    print("Received SIGTERM. Cleaning up and exiting...")
    cleanup_resources()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
with open("secrets.json", "r") as f:
    secrets = json.load(f)
OPENAI_API_KEY = secrets["OPENAI_API_KEY"]
header = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

# Environment-specific imports
try:
    from datasets import load_dataset
except ImportError:
    print("datasets module not found. Please install it if using the insta environment.")

try:
    from insta import InstaTransformersTool
except ImportError:
    print("InstaTransformersTool not found. Please ensure the insta module is available if using the insta environment.")

try:
    from desktop_env.desktop_env import DesktopEnv
    from parse_computer_use import parse_computer_use_pyautogui
except ImportError:
    print("DesktopEnv or parse_computer_use_pyautogui not found. Please ensure OSWorld dependencies are installed if using osworld environment.")

# --- System prompt constants ---
SYS_TASK_INIT = """What does this screen show? Imagine you are a real user on this webpage. Given the webpage link, please provide a single task that a user might perform on this website and the corresponding first action towards completing that task. You can use any software in the computer and the web. Be creative and come up with diverse tasks. The task should be simple enough that can be finished in a few steps. You should propose tasks that are clear and specific.

Task proposal rules:
1. The website should be explicitly mentioned in the task description.
2. The task should be specific, clear, and easily evaluated, not open-ended.
3. The task should be achievable within 1-3 steps.
4. The task should be relevant to the content of the webpage.
5. You should only propose tasks that do not require login to execute the task.
6. Generally try to avoid tasks that require sending emails, submitting forms, login, or other forms of communication.
7. Provide concrete information or constraints to the task, and use mock-up information (identifier, number, personal information, name, attributes, etc.) to make the task more specific and realistic.
8. The task description should provide all the necessary information to complete the task.
9. Do not propose tasks that are not possible in the Playwright API, such as downloading reports or opening and interacting with files.

Output with JSON block:
```{"task":"<TASK>", "action":"<ACTION>"}```
"""

SYS_TASK_INIT_LOCAL = """What does this screen show? Imagine you are a real user on this webpage. Given the software, please provide a single task that a user might perform using this software and the corresponding first action towards completing that task. You can use any tool in the computer and the web. Be creative and come up with diverse tasks. The task should be simple enough that can be finished in a few steps. You should propose tasks that are clear and specific.

Task proposal rules:
1. The software should be explicitly mentioned in the task description.
2. The task should be specific and clear.
3. The task should be achievable within a few steps.
4. The task should be relevant to the content of the webpage.
5. You should only propose tasks that do not require login to execute the task.
6. Provide concrete information or constraints to the task, and use mock-up information (identifier, number, personal information, name, attributes, etc.) to make the task more specific and realistic.
7. The task description should provide all the necessary information to complete the task.

Output with JSON block:
```{"task":"<TASK>", "action":"<ACTION>"}```
"""

SYS_TASK_FOLLOWUP = """What does this screen show? Imagine you are a real user on this webpage. Given the website link, and the tasks the user has done, please provide a single followup task that a user might perform on this website and the corresponding first action towards completing that task. You can use any software in the computer and the web. Be creative and come up with diverse tasks. The task should be simple enough that can be finished in a few steps.

Task proposal rules:
1. The website should be explicitly mentioned in the task description.
2. The task should depend on the previous tasks.
3. The task should be specific, clear, and easily evaluated, not open-ended.
4. The task should be achievable within 1-3 steps.
5. The task should be relevant to the content of the webpage.
6. You should only propose tasks that do not require login to execute the task.
7. Provide concrete information or constraints to the task, and use mock-up information (identifier, number, personal information, name, attributes, etc.) to make the task more specific and realistic.
8. The task description should provide all the necessary information to complete the task.
9. The task should be relevant to the overall task listed in the user prompt, if applicable. If the overall task is achievable within a few steps,
you can simply propose the overall task as the followup task. If the overall task is already achieved, you can propose an extension of the overall task.
10. Do not propose tasks that are not possible in the Playwright API, such as downloading reports or opening and interacting with files.
11. Try to avoid tasks that require sending emails, messages, submitting forms, login, or other forms of communication.
12. Avoid tasks that modify the backend state of the website.

Output with JSON block:
```{"task":"<TASK>", "action":"<ACTION>"}```
"""

SYS_TASK_FOLLOWUP_LOCAL = """What does this screen show? Imagine you are a real user on this computer. Given the software, and the tasks the user has done, please provide a single followup task that a user might perform on this computer and the corresponding first action towards completing that task. You can use any tool in the computer and the web. Be creative and come up with diverse tasks. The task should be simple enough that can be finished in a few steps.

Task proposal rules:
1. The task should depend on the previous tasks.
2. The task should be specific and clear.
3. The task should be achievable within a few steps.
4. The task should be relevant to the content of the software.
5. You should only propose tasks that do not require login to execute the task.
6. Provide concrete information or constraints to the task, and use mock-up information (identifier, number, personal information, name, attributes, etc.) to make the task more specific and realistic.
7. The task description should provide all the necessary information to complete the task.

Output with JSON block:
```{"task":"<TASK>", "action":"<ACTION>"}```
"""

SYS_TASK_ACTION = """You are an agent which follows my instruction and performs desktop computer tasks as instructed. You have good knowledge of computers and a good internet connection and assume your code will run on a computer controlling the mouse and keyboard. For each step, you will be asked to complete a task, and you will get an observation of an image, which is a screenshot of the computer screen. You also know the actions already taken towards the task. If you are on the lock screen, note that the password for the user is "password". You need to predict the next action based on the image. If similar actions have been tried several times but haven't succeeded, analyze the reason and propose a different action. If you think the task is finished, return "DONE" in the action field.
Rules:
1. First analyze the screenshot carefully, pay attention to details in the screenshot like popups, etc. 
2. Then analyze the previous thoughts and actions to make sure you are on the right track.
3. If you find you have tried similar actions several times but haven't success, analyze the reason carefully. Try to think if you were wrong or you missed any steps. Carefully analyze the screenshot to find the reason of the failure and propose a different action.
4. If you think the task is finished, return "DONE" in the action field.
5. Do not try actions that are not accessible in the Playwright API, like downloading files.
6. Do not send messages to websites, or try to login into servers.

Output with JSON block:
```{"thoughts":"<THOUGHTS and REASONS>", "action":"<ACTION>"}```
"""

with open("sys_task_action_no_done.txt", "r") as f:
    SYS_TASK_WEB_ACTION = f.read()

SYS_COMPUTER_ACTION = """You are an agent which follow my instruction and perform desktop computer tasks as instructed.
You have good knowledge of computer and good internet connection and assume your code will run on a computer for controlling the mouse and keyboard.
For each step, you will be asked to finish a task, and you will get an observation of an image, which is the screenshot of the computer screen. 
You need to predict the next action of the computer based on the image. You have all the permissions to proceed and you don't need to ask for permission. 
The safety checks are acknowledged and you can always proceed."""

SYS_TASK_SUMMARIZE = """
You are a helpful assistant that summarizes task sequences.
Given a sequence of tasks and actions that solve them, summarize the sequence into a single task and a sequence of actions that solve it.
Format your response as a JSON object, and only this JSON object, with the following fields:
- task: The summarized task
- website: The website that the task is performed on
- action_sequence: The sequence of actions that solve the task
- thoughts_sequence: The sequence of thoughts that corresponded to each action
If possible, try to make your summary task not open-ended.
If you believe that the agent did not actually succeed in the task, please ignore it.
If the agent did not succeed in any task, return "FAILED" in the task field.
Try to incorporate every relevant task in your summarized task. Your summarized task can be multi-part and 
have multiple sentences.
Try to include all actions in your summarized action sequence.
"""

SYS_PROMPTS_ACTION = {
    'insta': SYS_TASK_WEB_ACTION,
    'osworld': SYS_TASK_ACTION
}

SYS_TASK_DONE = """
You are a helpful assistant that determines if a task is done.
Given a task, observation, previous thoughts, and previous actions, determine if the task is done.
Output with JSON block:
```{"thoughts": "<THOUGHTS>", "is_done": <BOOLEAN>}```
"""

# --- Helper functions for interacting with the language models ---
def call_gpt_4o(sys_prompt, user_prompt, model='gpt-4o', img=None, max_retries=3):
    """
    Call the OpenAI GPT-4o API with the given prompts and optional image.
    
    Args:
        sys_prompt (str): The system prompt to provide context to the model.
        user_prompt (str): The user prompt containing the query or instruction.
        model (str, optional): The model to use. Defaults to 'gpt-4o'.
        img (str, optional): Base64-encoded image to include with the prompt. Defaults to None.
        max_retries (int, optional): Maximum number of retries for API calls. Defaults to 3.
        
    Returns:
        str: The text response from the model.
        
    Raises:
        Exception: If the API call fails after max_retries attempts.
    """
    for attempt in range(max_retries):
        try:
            payload = {
                "model": model,
                "temperature": 1.0,
                "truncation": "auto",
                "input": []
            }
            payload["input"].append({"role": "system", "content": [{"type": "input_text", "text": sys_prompt}]})
            user_content = [{"type": "input_text", "text": user_prompt}]
            if img:
                user_content.append({"type": "input_image", "image_url": f"data:image/png;base64,{img}"})
            payload["input"].append({"role": "user", "content": user_content})
            
            response = requests.post("https://api.openai.com/v1/responses", headers=header, json=payload)
            
            if response.status_code != 200:
                print(f"API error: {response.status_code} - {response.text}")
                time.sleep(2)  # Backoff before retry
                continue
                
            response_json = response.json()
            if 'output' not in response_json or len(response_json['output']) == 0:
                print(f"Unexpected API response format: {response_json}")
                time.sleep(2)
                continue
                
            return response_json['output'][0]['content'][0]['text']
            
        except requests.exceptions.RequestException as e:
            print(f"Request error (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2)
        except (KeyError, IndexError) as e:
            print(f"Response parsing error (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2)
    
    raise Exception(f"Failed to get response from {model} after {max_retries} attempts")

def call_o3(sys_prompt, user_prompt, model='o3-mini', img=None):
    """
    Call the OpenAI o3 API with the given prompts and optional image.
    
    Args:
        sys_prompt (str): The system prompt to provide context to the model.
        user_prompt (str): The user prompt containing the query or instruction.
        model (str, optional): The model to use. Defaults to 'o3-mini'.
        img (str, optional): Base64-encoded image to include with the prompt. Defaults to None.
        
    Returns:
        str: The text response from the model.
    """
    payload = {
        "model": model,
        "temperature": 1.0,
        "truncation": "auto",
        "input": []
    }
    payload["input"].append({"role": "system", "content": [{"type": "input_text", "text": sys_prompt}]})
    user_content = [{"type": "input_text", "text": user_prompt}]
    if img:
        user_content.append({"type": "input_image", "image_url": f"data:image/png;base64,{img}"})
    payload["input"].append({"role": "user", "content": user_content})
    response = requests.post("https://api.openai.com/v1/responses", headers=header, json=payload)
    return response.json()['output'][1]['content'][0]['text']

def call_computer_use_preview(sys_prompt, user_prompt, img):
    payload = {
        "model": "computer-use-preview", 
        "temperature": 1.0, 
        'truncation': 'auto',
        'tools': [{
            "type": "computer_use_preview",
            "display_width": 1920,
            "display_height": 1080,
            "environment": "windows" # other possible values: "mac", "windows", "linux"
        }],
        'reasoning': {'effort': 'medium', 'generate_summary': 'concise'},
        "input": [
            {
                "role": "system", 
                "content": [
                    {
                        "type": "input_text", 
                        "text": sys_prompt
                        }
                ]
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "input_text", 
                        "text": user_prompt
                    },
                    {
                        "type": "input_image", 
                        "image_url": f"data:image/png;base64,{img}"
                    } 
                ]
            }
        ]
    }
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers=header,
        json=payload
    )
    # breakpoint()

    try:
        return response.json()['output']
    except:
        raise Exception("No computer call found")

def parse_json(llm_output):
    """
    Parse JSON from LLM output text, handling various formats and edge cases.
    
    Args:
        llm_output (str): The raw text output from an LLM that contains JSON.
        
    Returns:
        dict: The parsed JSON object, or None if parsing fails.
    """
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(\{.*?\})```", llm_output, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        match = re.search(r"(\{.*\})", llm_output, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            print("No JSON object found in:", llm_output)
            return None
    try:
        data = json.loads(json_str)
        if isinstance(data.get('action'), str) and data['action'] != 'DONE':
            try:
                action_str = data['action'].replace('\"', '"')
                if action_str.startswith('"') and action_str.endswith('"'):
                    action_str = action_str[1:-1]
                data['action'] = json.loads(action_str)
            except json.JSONDecodeError:
                pass
        return data
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        return None

# --- Task proposal functions for the InSTA (web) environment ---
def initial_task_propose(website, webpage_text, model='gpt-4o', max_tries=5):
    """
    Generate an initial task proposal for a given website.
    
    Args:
        website (str): The website domain to generate a task for.
        webpage_text (str): The text content of the webpage.
        model (str, optional): The model to use for task generation. Defaults to 'gpt-4o'.
        max_tries (int, optional): Maximum number of attempts to generate a valid task. Defaults to 5.
        
    Returns:
        str: The proposed task.
        
    Raises:
        Exception: If task generation fails after max_tries attempts.
    """
    sys_prompt = SYS_TASK_INIT
    user_prompt = f"Given the website: {website} and webpage layout {webpage_text}, \nplease propose a task."
    print("Generating artificial initial task")
    for attempt in range(max_tries):
        try:
            if model.startswith('gpt'):
                llm_output = call_gpt_4o(sys_prompt, user_prompt, model=model)
            else:
                llm_output = call_o3(sys_prompt, user_prompt, model=model)
            data = parse_json(llm_output)
            task_info = data.get('proposed_task') or data.get('task')
            if not task_info:
                raise Exception("No task found")
            return task_info
        except Exception as e:
            print(f"Error in initial task proposal: {e}. Retrying...")
            time.sleep(1)
    raise Exception("Failed to generate initial task")

def followup_task_propose(website, webpage_text, task_history, model='gpt-4o', max_tries=5, overall_task=""):
    """
    Generate a followup task proposal based on previous tasks and current webpage state.
    
    Args:
        website (str): The website domain to generate a task for.
        webpage_text (str): The text content of the webpage.
        task_history (list): List of previously completed tasks.
        model (str, optional): The model to use for task generation. Defaults to 'gpt-4o'.
        max_tries (int, optional): Maximum number of attempts to generate a valid task. Defaults to 5.
        overall_task (str, optional): An overall task that the followup should progress towards. Defaults to "".
        
    Returns:
        str: The proposed followup task.
        
    Raises:
        Exception: If task generation fails after max_tries attempts.
    """
    sys_prompt = SYS_TASK_FOLLOWUP
    user_prompt = f"Given the website: {website}, webpage layout: {webpage_text} \nand task history: {task_history}, please propose a followup task."
    print("Generating artificial followup task")
    if overall_task:
        user_prompt += f" Overall task: {overall_task}."
    for attempt in range(max_tries):
        try:
            if model.startswith('gpt'):
                llm_output = call_gpt_4o(sys_prompt, user_prompt, model=model)
            else:
                llm_output = call_o3(sys_prompt, user_prompt, model=model)
            data = parse_json(llm_output)
            task_info = data.get('proposed_task') or data.get('task')
            return task_info
        except Exception as e:
            print(f"Error in followup task proposal: {e}. Retrying...")
            time.sleep(1)
    raise Exception("Failed to generate followup task")

# --- Task proposal functions for the OSWorld (local) environment ---
def initial_task_propose_local(software, base64_image, model='gpt-4o', max_retries=5):
    """
    Generate an initial task proposal for local software in the OSWorld environment.
    
    Args:
        software (str): The name of the software to generate a task for.
        base64_image (str): Base64-encoded screenshot of the current state.
        model (str, optional): The model to use for task generation. Defaults to 'gpt-4o'.
        max_retries (int, optional): Maximum number of retries. Defaults to 5.
        
    Returns:
        str: The proposed task.
        
    Raises:
        Exception: If task generation fails after max_retries attempts.
    """
    sys_prompt = SYS_TASK_INIT_LOCAL
    user_prompt = f"Given the software {software}, what task would a user perform?"
    
    for attempt in range(max_retries):
        try:
            llm_output = call_gpt_4o(sys_prompt, user_prompt, model=model, img=base64_image)
            data = parse_json(llm_output)
            task_info = data.get('task')
            if task_info:
                return task_info
        except Exception as e:
            print(f"Error in initial_task_propose_local (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(1)
    
    raise Exception(f"Failed to generate initial task after {max_retries} attempts")

def followup_task_propose_local(software, task_history, base64_image, model='gpt-4o', max_retries=5):
    """
    Generate a followup task proposal for local software in the OSWorld environment.
    
    Args:
        software (str): The name of the software to generate a task for.
        task_history (list): List of previously completed tasks.
        base64_image (str): Base64-encoded screenshot of the current state.
        model (str, optional): The model to use for task generation. Defaults to 'gpt-4o'.
        max_retries (int, optional): Maximum number of retries. Defaults to 5.
        
    Returns:
        str: The proposed followup task.
        
    Raises:
        Exception: If task generation fails after max_retries attempts.
    """
    sys_prompt = SYS_TASK_FOLLOWUP_LOCAL
    user_prompt = f"Given the software {software} and the task history {task_history}, what would be a followup task?"
    
    for attempt in range(max_retries):
        try:
            llm_output = call_gpt_4o(sys_prompt, user_prompt, model=model, img=base64_image)
            data = parse_json(llm_output)
            task_info = data.get('task')
            if task_info:
                return task_info
        except Exception as e:
            print(f"Error in followup_task_propose_local (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(1)
    
    raise Exception(f"Failed to generate followup task after {max_retries} attempts")

# --- Action generation and computer-use action functions ---
def generate_action(task, thoughts_history, action_history, observation, model='gpt-4o', max_tries=5, max_steps=10,
                    env_type='osworld', base64_image=None, obs_max_len=1000):
    """
    Generate the next action to take given the current task and state.
    
    Args:
        task (str): The current task to complete.
        thoughts_history (list): History of previous thoughts.
        action_history (list): History of previous actions.
        observation (str): Text observation of the current state (webpage text or a11y tree).
        model (str, optional): The model to use for action generation. Defaults to 'gpt-4o'.
        max_tries (int, optional): Maximum number of attempts to generate a valid action. Defaults to 5.
        max_steps (int, optional): Maximum number of steps allowed for the task. Defaults to 10.
        base64_image (str, optional): Base64-encoded screenshot for image-based environments. Defaults to None.
        
    Returns:
        tuple: (action, thoughts) where action is the next action to take and thoughts is the reasoning.
    """
    # observation will be webpage_text in InSTA or the accessibility tree (a11y_tree) in OSWorld (defunct).
    # base64_image is the screenshot of the current state of the computer in OSWorld.
    print("Observation:", observation[:obs_max_len])
    sys_prompt = SYS_PROMPTS_ACTION[env_type]
    user_prompt = f"Given task: {task}\nObservation: {observation}\nPrevious thoughts: {thoughts_history}\nPrevious actions: {action_history}\nWhat is the next action? If complete, return DONE."
    for attempt in range(max_tries):
        try:
            if model.startswith('gpt'):
                llm_output = call_gpt_4o(sys_prompt, user_prompt, model=model, img=base64_image)
            else:
                llm_output = call_o3(sys_prompt, user_prompt, model=model, img=base64_image)
            data = parse_json(llm_output)
            if data is None:
                raise Exception("Failed to parse JSON response")
            action_info = data.get('action')
            # breakpoint()
            thoughts = data.get('thoughts', "No thoughts provided")
            if action_info is None:
                raise Exception("No action found in response")
            if env_type == 'insta' and action_info != 'DONE' and isinstance(action_info, str):
                try:
                    json.loads(action_info)
                except:
                    raise Exception("Invalid action format")
            return action_info, thoughts
        except Exception as e:
            print(f"Error in generating action: {e}. Retrying...")
            time.sleep(2)
    print("All attempts failed. Using default action.")
    return str({"action_key": "click", "action_kwargs": {}, "target_element_id": "1109"}), "Default action"

def is_task_done(task, thoughts_history, action_history, observation, model='gpt-4o-mini', max_tries=5):
    """
    Determine if a task is completed based on the current state.
    
    Args:
        task (str): The current task to check.
        thoughts_history (list): History of previous thoughts.
        action_history (list): History of previous actions.
        observation (str): Text observation of the current state.
        model (str, optional): The model to use. Defaults to 'gpt-4o-mini'.
        max_tries (int, optional): Maximum number of attempts. Defaults to 5.
        
    Returns:
        bool: True if the task is completed, False otherwise.
    """
    sys_prompt = SYS_TASK_DONE
    user_prompt = f"Given task: {task}\nObservation: {observation}\nPrevious thoughts: {thoughts_history}\nPrevious actions: {action_history}\nIs the task done?"
    
    # If the observation indicates an error, the task cannot be done
    if "Error starting environment" in observation or "Failed to execute action" in observation:
        print("Task cannot be completed due to environment error")
        return False
        
    for attempt in range(max_tries):
        try:
            llm_output = call_gpt_4o(sys_prompt, user_prompt, model=model)
            data = parse_json(llm_output)
            if data is None:
                raise Exception("Failed to parse is_done response")
            print("Judge thoughts:", data.get('thoughts', "No thoughts provided"))
            return data.get('is_done', False)
        except Exception as e:
            print(f"Error in is_task_done: {e}. Retrying...")
            time.sleep(1)
    return False

def generate_computer_use_action(task, step, command_history, base64_image, iter=0, max_iter=3, model='o3-mini'):
    """
    Generate a computer use action (PyAutoGUI command) for the OSWorld environment.
    
    Args:
        task (str): The current task to complete.
        step (str): The current step description.
        command_history (list): History of previous commands.
        base64_image (str): Base64-encoded screenshot of the current state.
        iter (int, optional): Current iteration count for retries. Defaults to 0.
        max_iter (int, optional): Maximum number of retry iterations. Defaults to 3.
        model (str, optional): The model to use for action generation. Defaults to 'o3-mini'.
        
    Returns:
        str: A PyAutoGUI command to execute.
        
    Raises:
        Exception: If action generation fails after max_iter attempts.
    """
    sys_prompt = SYS_COMPUTER_ACTION
    user_prompt = f"Given the task: {task}, I have done the following actions: {command_history}. Next, I need to do the step: {step}. What would be the action?"
    # breakpoint()
    if iter < max_iter:
        try:
            llm_output = call_computer_use_preview(sys_prompt, user_prompt, base64_image)
            action_info = None
            for i in range(len(llm_output)):
                if llm_output[i]['type'] == 'computer_call':
                    action_info = llm_output[i]['action']
                    break
            if action_info is None:
                raise Exception("Action in computer call not found")
            action_info = parse_computer_use_pyautogui(action_info)
            if action_info is None:
                raise Exception("Action was not parsed")
            return action_info
        except Exception as e:
            print(f"Error in generate_computer_use_action: {e}. Retrying...")
            return generate_computer_use_action(task, step, command_history, base64_image, iter+1, max_iter, model)
    else:
        raise Exception(f"Failed to generate computer use action after {max_iter} attempts")

def encode_image_from_variable(image_content):
    """
    Encode binary image content to base64 string.
    
    Args:
        image_content (bytes): Binary image content.
        
    Returns:
        str: Base64-encoded image string.
    """
    return base64.b64encode(image_content).decode('utf-8')

# --- Task solving functions ---
# For OSWorld (image-based) environment:
def solve_task_osworld(task, env, base64_image, a11y_tree, time_btw_tasks=5, max_steps=15):
    """
    Solve a task in the OSWorld (local desktop) environment.
    
    Args:
        task (str): The task to complete.
        env: The OSWorld environment object.
        base64_image (str): Base64-encoded screenshot of the current state.
        a11y_tree (str): Accessibility tree of the current state.
        time_btw_tasks (int, optional): Time to wait between actions. Defaults to 5.
        max_steps (int, optional): Maximum number of steps to attempt. Defaults to 15.
        
    Returns:
        bool: True if the task was completed successfully, False otherwise.
    """
    thoughts_history = []
    action_history = []
    command_history = []
    for i in tqdm(range(max_steps), desc="Solving Task"):
        action, thoughts = generate_action(task, thoughts_history, action_history, "", env_type='osworld',
                                          model='gpt-4o', base64_image=base64_image)
        # a11y_tree is often too large to be included in the observation, so we don't include it.
        thoughts_history.append(thoughts)
        action_history.append(action)
        if action == 'DONE':
            print("Task completed.")
            return True, thoughts_history, command_history
        python_command = generate_computer_use_action(task, action, command_history, base64_image)
        command_history.append(python_command)
        python_command += f'; time.sleep({time_btw_tasks})'
        print("Thoughts:", thoughts)
        print("Action:", action)
        print("Executing command:", python_command)
        obs, reward, done, info = env.step(python_command)
        base64_image = encode_image_from_variable(obs['screenshot'])
        a11y_tree = obs['accessibility_tree']
    return False, thoughts_history, command_history

# For InSTA (web-based) environment:
def setup_runpod_endpoint(endpoint_url=None, api_key=None):
    """
    Set up connection to a RunPod serverless endpoint running the insta-browser-environment.
    
    Args:
        endpoint_url (str, optional): The URL of your RunPod serverless endpoint. If None, reads from secrets.
        api_key (str, optional): Your RunPod API key. If None, reads from secrets.
        
    Returns:
        bool: True if the endpoint is reachable, False otherwise
    """
    global RUNPOD_ENDPOINT_URL, RUNPOD_API_KEY
    
    # Use provided values or read from secrets
    if endpoint_url is None or api_key is None:
        if "RUNPOD_ENDPOINT_URL" not in secrets or "RUNPOD_API_KEY" not in secrets:
            print("RunPod endpoint URL and API key must be provided in secrets.json")
            return False
        
        RUNPOD_ENDPOINT_URL = secrets.get("RUNPOD_ENDPOINT_URL")
        RUNPOD_API_KEY = secrets.get("RUNPOD_API_KEY")
    else:
        RUNPOD_ENDPOINT_URL = endpoint_url
        RUNPOD_API_KEY = api_key
    
    # Test the connection
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RUNPOD_API_KEY}"
        }
        
        # Simple health check request
        response = requests.get(
            f"{RUNPOD_ENDPOINT_URL}/health",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print("Successfully connected to RunPod endpoint")
            return True
        else:
            print(f"RunPod endpoint returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to RunPod endpoint: {e}")
        return False

def extract_session_id(outputs):
    """Extract session ID from tool outputs."""
    if "session ID:" in outputs:
        try:
            return outputs.split("session ID: `")[1].split("`")[0]
        except IndexError:
            return None
    return None

def extract_webpage_text(outputs):
    """Extract webpage text from tool outputs."""
    if "in markdown:" in outputs:
        return outputs.split("in markdown:")[1].strip()
    return ""

def is_error_response(outputs):
    """Check if the response contains an error."""
    error_indicators = [
        "Error starting environment",
        "Failed to execute action",
        "The provided action could not be parsed",
        "Failed to load URL",
        "TimeoutError"
    ]
    return any(indicator in outputs for indicator in error_indicators)

def is_navigation_action(action):
    """Check if an action is a navigation action and extract the URL."""
    try:
        action_obj = json.loads(action) if isinstance(action, str) else action
        if action_obj.get("action_key") == "goto" and "url" in action_obj.get("action_kwargs", {}):
            return True, action_obj["action_kwargs"]["url"]
    except:
        pass
    return False, None

def solve_task_insta(tool, task, website, outputs=None, model='gpt-4o', max_steps=10):
    """
    Solve a task in the InSTA (web-based) environment.
    
    Args:
        tool: The InSTA tool object.
        task (str): The task to complete.
        website (str): The website domain.
        outputs (str, optional): Initial outputs from the tool. Defaults to None.
        model (str, optional): The model to use for action generation. Defaults to 'gpt-4o'.
        max_steps (int, optional): Maximum number of steps to attempt. Defaults to 10.
        
    Returns:
        tuple: (completed, webpage_text, action_history, thoughts_history) where completed is a boolean
               indicating if the task was completed successfully.
    """
    thoughts_history = []
    action_history = []
    successful_actions = []  # Track all successful actions
    
    # Checkpoint system
    checkpoints = []
    current_checkpoint = {
        "url": "http://" + website,
        "actions": []  # Actions since this checkpoint
    }
    
    # Track if we're in a recovery state
    recovery_mode = False
    last_error = ""
    consecutive_errors = 0
    max_consecutive_errors = 3  # Maximum allowed consecutive errors before forcing a reset
    
    # Initialize the environment
    if outputs is None:
        outputs = tool(url=current_checkpoint["url"])
    
    # Check for environment errors
    if is_error_response(outputs):
        print(f"Environment error detected: {outputs}")
        thoughts = "The environment is currently not starting, which means I cannot complete the task at this moment."
        thoughts_history.append(thoughts)
        action_history.append("DONE")
        # More robust solution TODO
        return False, extract_webpage_text(outputs), action_history, thoughts_history
        
    session_id = extract_session_id(outputs)
    
    print(f"Starting to solve task: {task}")
    print(f"Session ID: {session_id}")
    print(f"Initial URL: {current_checkpoint['url']}")
    
    for i in tqdm(range(max_steps), desc="Solving Task"):
        webpage_text = extract_webpage_text(outputs)
        print(f"Step {i+1}/{max_steps}: Webpage length: {len(webpage_text)} characters")
            
        # Check for execution errors
        if is_error_response(outputs):
            print(f"Action execution error detected: {outputs}")
            recovery_mode = True
            last_error = outputs
            consecutive_errors += 1
            print(f"Consecutive errors: {consecutive_errors}/{max_consecutive_errors}")
            
            # If we've hit too many consecutive errors, force a complete reset
            if consecutive_errors >= max_consecutive_errors:
                print(f"Too many consecutive errors ({consecutive_errors}). Forcing complete reset.")
                outputs = tool(url="http://" + website)
                
                if not is_error_response(outputs):
                    print("Fresh session started successfully after forced reset.")
                    checkpoints = []
                    current_checkpoint = {"url": "http://" + website, "actions": []}
                    session_id = extract_session_id(outputs)
                    consecutive_errors = 0  # Reset error counter
                    print(f"New session ID: {session_id}")
                else:
                    print("Even fresh session failed. Task cannot be completed.")
                    return False, webpage_text, action_history, thoughts_history
            else:
                # Attempt to recover using the checkpoint system
                if checkpoints:
                    print(f"Attempting to recover using checkpoint system. We have {len(checkpoints)} checkpoints.")
                    restored = restore_from_checkpoints(tool, checkpoints)
                    
                    if restored:
                        outputs, session_id, current_checkpoint = restored
                        print(f"Successfully restored session. New session ID: {session_id}")
                        consecutive_errors = 0  # Reset error counter after successful recovery
                    else:
                        # Start fresh if restoration failed
                        print("Failed to restore using any checkpoint. Starting fresh session.")
                        outputs = tool(url="http://" + website)
                        
                        if not is_error_response(outputs):
                            print("Fresh session started successfully.")
                            checkpoints = []
                            current_checkpoint = {"url": "http://" + website, "actions": []}
                            session_id = extract_session_id(outputs)
                            consecutive_errors = 0  # Reset error counter
                            print(f"New session ID: {session_id}")
                else:
                    # No checkpoints available, start fresh
                    print("No checkpoints available. Starting fresh session.")
                    outputs = tool(url="http://" + website)
                    
                    if not is_error_response(outputs):
                        print("Fresh session started successfully.")
                        current_checkpoint = {"url": "http://" + website, "actions": []}
                        session_id = extract_session_id(outputs)
                        consecutive_errors = 0  # Reset error counter
                        print(f"New session ID: {session_id}")
            
        # Modify the user prompt to include error information if in recovery mode
        user_prompt_addition = ""
        if recovery_mode:
            user_prompt_addition = f"\nThe previous action failed with error: {last_error[:200]}... Please try a different approach."
            recovery_mode = False  # Reset recovery mode after informing the agent
            
        # Generate the next action
        action, thoughts = generate_action(task, thoughts_history, action_history, 
                                          webpage_text + user_prompt_addition, 
                                          env_type='insta', model=model, max_steps=max_steps)
        
        if action == 'DONE':
            # Check if the task is really done
            is_done = is_task_done(task, thoughts_history, action_history, webpage_text, model=model)
            if is_done:
                thoughts_history.append(thoughts)
                action_history.append(action)
                print(f"Task completed in {i+1} steps.")
                return True, webpage_text, action_history, thoughts_history
            else:
                print("Task was marked as done but the judge disagreed. Retrying...")
                continue
            
        # Convert action to string if it's a dictionary
        if isinstance(action, dict):
            action = json.dumps(action)
            
        print(f"Step {i+1}/{max_steps} - Action: {action}\nThoughts: {thoughts[:100]}...\n")
        print(f"Executing action: {action[:100]}...")
        
        # Check if this is a navigation action
        is_navigation, new_url = is_navigation_action(action)
        if is_navigation:
            print(f"Navigation action detected. New URL: {new_url}")
        
        # Execute the action
        temp_outputs = tool(session_id=session_id, action=action) 
        print(f"Received response of length: {len(temp_outputs)} characters")
        
        # Update session ID if it changed
        new_session_id = extract_session_id(temp_outputs)
        if new_session_id and new_session_id != session_id:
            print(f"Session ID changed from {session_id} to {new_session_id}")
            session_id = new_session_id
        
        # Handle action execution failures
        if is_error_response(temp_outputs):
            print(f"Action execution failed: {temp_outputs}")
            outputs = temp_outputs
            recovery_mode = True
            last_error = temp_outputs
            continue
        
        # If we get here, the action was successful - add to histories
        thoughts_history.append(thoughts)
        action_history.append(action)
        successful_actions.append(action)
        outputs = temp_outputs
        consecutive_errors = 0  # Reset consecutive error counter on success
        
        # Update checkpoint system
        if is_navigation and new_url:
            # Save the current checkpoint before creating a new one
            checkpoints.append(current_checkpoint)
            # Create a new checkpoint at this navigation point
            current_checkpoint = {"url": new_url, "actions": []}
            print(f"Created new checkpoint at URL: {new_url}")
        else:
            # Add this action to the current checkpoint's action list
            current_checkpoint["actions"].append(action)

    print(f"Task did not complete within {max_steps} steps.")
    return False, webpage_text, action_history, thoughts_history

# Add this function to your script
def create_openai_browser_agent(model="gpt-4o", api_key=None, client_kwargs=None):
    """
    Create a BrowserAgent configured to use OpenAI API.
    
    Args:
        model (str): The OpenAI model to use (default: "gpt-4o")
        api_key (str, optional): OpenAI API key. If None, uses the one from secrets.
        client_kwargs (dict, optional): Additional kwargs for the client configuration.
        
    Returns:
        BrowserAgent: Configured browser agent
    """
    from insta.configs.agent_config import get_agent_config
    from insta import BrowserAgent
    
    # Use the API key from secrets if not provided
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    if client_kwargs is None:
        client_kwargs={
            "api_key": api_key,
            # No base_url needed for OpenAI (it uses the default)
        }

    # Configure the agent to use OpenAI API
    agent_config = get_agent_config(
        # Use OpenAI's tokenizer (or another appropriate one)
        tokenizer="gpt2",  
        # Configure OpenAI client
        client_kwargs=client_kwargs,
        
        # Configure generation parameters
        generation_kwargs={
            "model": model,
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 1.0
        },
        
        # Other parameters
        # max_history=5,
        max_obs_tokens=4096,
        catch_errors=True,
        max_errors=5,
        log_errors=True
    )
    
    # Create the agent with JSON action parser
    return BrowserAgent(config=agent_config)

def create_openai_browser_judge(model="gpt-4o", api_key=None, client_kwargs=None):
    """
    Create a BrowserJudge configured to use OpenAI API.
    
    Args:
        model (str): The OpenAI model to use (default: "gpt-4o")
        api_key (str, optional): OpenAI API key. If None, uses the one from secrets.
        client_kwargs (dict, optional): Additional kwargs for the client configuration.
        
    Returns:
        BrowserJudge: Configured browser judge
    """
    from insta.configs.judge_config import get_judge_config
    from insta import BrowserJudge
    
    # Use the API key from secrets if not provided
    if api_key is None:
        api_key = OPENAI_API_KEY
    if client_kwargs is None:
        client_kwargs={
            "api_key": api_key,
            # No base_url needed for OpenAI (it uses the default)
        }
    
    # Configure the judge to use OpenAI API
    judge_config = get_judge_config(
        tokenizer="gpt2",  
        client_kwargs=client_kwargs,
        generation_kwargs={
            "model": model,
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 1.0
        },
        max_obs_tokens=4096,
        catch_errors=True,
        max_errors=5,
        log_errors=True
    )
    
    # Create the judge with JSON judgment parser
    return BrowserJudge(config=judge_config)

def ensure_context_sync(agent):
    """
    Checks if the agent's context is in a valid state and attempts to fix it if not.
    
    Args:
        agent: The BrowserAgent instance to check and fix
        
    Returns:
        bool: True if context is valid or was fixed, False if it couldn't be fixed
    """
    observations, instructions, urls, actions = agent.get_context()
    
    valid_context = (
        len(observations) == (len(actions) + 1) and 
        len(observations) == len(urls) and 
        len(observations) == len(instructions)
    )
    
    if valid_context:
        return True
    
    print("Warning: Agent context is out of sync. Attempting to fix...")
    
    # Case 1: Too many actions
    if len(actions) >= len(observations):
        # Remove excess actions
        actions = actions[:len(observations)-1] if len(observations) > 0 else []
        print(f"Fixed: Removed excess actions. Now have {len(actions)} actions and {len(observations)} observations")
    
    # Case 2: Mismatched observation-related lists
    min_obs_len = min(len(observations), len(urls), len(instructions))
    if min_obs_len > 0:
        # Trim all observation-related lists to the same length
        observations = observations[:min_obs_len]
        urls = urls[:min_obs_len]
        instructions = instructions[:min_obs_len]
        
        # Ensure actions list is exactly one shorter
        if len(actions) > min_obs_len - 1:
            actions = actions[:min_obs_len - 1]
        
        print(f"Fixed: Aligned context lists to consistent lengths. Observations: {len(observations)}, Actions: {len(actions)}")
        
        # Update the agent's context
        agent.set_context((observations, instructions, urls, actions))
        return True
    
    # If we get here, context is empty or severely broken
    print("Error: Could not fix agent context. Resetting...")
    agent.reset()
    return False

def solve_task_insta_with_agent(task, website, model='gpt-4o', max_steps=10, env: InstaEnv | None = None, previous_actions: list | None = None,
                                success_threshold = 0.9, client_kwargs=None, **kwargs):
    """
    Solve a task using BrowserAgent directly instead of the tool interface.
    Judge after each step to determine if the task is complete.
    
    Args:
        task (str): The task to solve
        website (str): The website URL
        model (str): The model to use
        max_steps (int): Maximum number of steps
        env (InstaEnv, optional): Existing environment to use. If None, creates a new one.
        previous_actions (list, optional): Previous successful actions to replay before starting new actions.
        client_kwargs (dict, optional): Additional kwargs for the client configuration.
        
    Returns:
        tuple: (success, final_observation, action_history, thoughts_history, judgment_history, env)
    """
    # Create environment, agent, and judge if not provided
    if env is None:
        try:
            browser_config = get_browser_config(
                playwright_port = 3000 + random.randint(0, 8),
            )
            env = InstaEnv(config=browser_config)
        except ConnectionRefusedError as e:
            print(f"Connection refused error: {e}")
            print("The InSTA browser environment server is not running or not accessible.")
            print("Please ensure the Docker container is running with ports 3000-3008 exposed.")
            # Return failure with appropriate error messages
            return False, "Connection error: Browser environment not accessible", [], ["Connection refused error. The browser environment server is not running."], [], None
        except Exception as e:
            print(f"Error creating environment: {e}")
            return False, f"Environment error: {str(e)}", [], [f"Failed to create environment: {str(e)}"], [], None
    
    try:
        agent = create_openai_browser_agent(model=model, client_kwargs=client_kwargs)
        judge = create_openai_browser_judge(model=model, client_kwargs=client_kwargs)
    except Exception as e:
        print(f"Error creating agent or judge: {e}")
        return False, f"Agent creation error: {str(e)}", [], [f"Failed to create agent or judge: {str(e)}"], [], env
    
    # Initialize histories
    action_history = []
    thoughts_history = []
    processed_text_history = []
    judgment_history = []
    
    # Ensure website has http:// prefix
    if not website.startswith("http"):
        website = "http://" + website
    
    # Reset environment with the website if no previous actions
    if previous_actions is None or len(previous_actions) == 0:
        observation, info = env.reset(url=website)
    else:
        # Use the existing environment state
        # Get the current observation
        observation = env.get_obs()
    
    # Store initial observation
    processed_text_history.append(observation.processed_text)
    
    # Replay previous actions if provided
    if previous_actions and len(previous_actions) > 0:
        print(f"Replaying {len(previous_actions)} previous actions...")
        for i, prev_action in enumerate(previous_actions):
            print(f"Replaying action {i+1}/{len(previous_actions)}")
            
            # Take step in environment with previous action
            observation, reward, done, truncated, info = env.step(action=prev_action)
            
            # Store in history
            # action_history.append(prev_action)
            # processed_text_history.append(observation.processed_text)
            
            # # If environment signals done during replay, we're finished
            # if done:
            #     print(f"Environment signaled completion during replay after {i+1} steps.")
            #     success = True
            #     return success, observation.processed_text, action_history, thoughts_history, judgment_history, env
    
    # Main interaction loop for new actions
    for i in tqdm(range(1, max_steps+1), desc="Solving Task"):
        step_num = i + len(previous_actions) if previous_actions else i
        print(f"Step {step_num}/{max_steps + len(previous_actions) if previous_actions else max_steps}")
        
        # Get action from agent
        action = NULL_ACTION
        while action == NULL_ACTION:
            action = agent(
                observation=observation.processed_text,
                instruction=task,
                current_url=website
            )
            if action == NULL_ACTION:
                print(f"ValueError in agent call, possibly due to context being out of sync.")
                # Try to fix the agent's context
                if ensure_context_sync(agent):
                    print("Agent context fixed, retrying...")
                    continue
                else:
                    print("Could not fix agent context. Resetting agent...")
                    agent.reset()
                    # Push the current observation again after reset
                    agent.push_observation(
                        observation=observation.processed_text,
                        instruction=task,
                        current_url=website
                    )
                    continue
            
            if action == NULL_ACTION:
                print(f"Action could not be processed from LLM output: {action}.")
        
        # Store action in history
        action_history.append(action)
        thoughts_history.append(action.response)
        print("Action:", action.matched_response)
        print("Action thoughts:", action.response[:1000])
        
        # Take step in environment
        try:
            observation, reward, done, truncated, info = env.step(action=action)
            
            # Only push the action to the agent's context if the step was successful
            agent.push_action(action.response)
            
            # Store observation
            processed_text_history.append(observation.processed_text)
            
        except Exception as e:
            print(f"Error executing action: {e}")
            # Don't push the action to agent context if it failed
            # Remove the action from our tracking history too
            action_history.pop()
            thoughts_history.pop()
            continue
        
        # Check for errors in the observation
        if is_error_response(observation.processed_text) or truncated:
            print(f"Invalid action detected in step {step_num}.")
            print(f"Observation: {observation.processed_text[:10000]}...")
            
            # Remove the failed action from agent's context
            agent.pop_action()
            
            # Remove from our tracking history too
            action_history.pop()
            thoughts_history.pop()
            processed_text_history.pop()
            
            # Reset environment and replay successful actions
            # Undo previous action TODO
            joined_actions = (previous_actions.copy() if previous_actions is not None else []) + action_history.copy()
            attempts = [0] * len(joined_actions)
            MAX_ATTEMPTS = 3
            successful_replay = False
            while not successful_replay:
                observation, info = env.reset(url=website)
                for i in range(len(joined_actions)):
                    print(f"Redoing action {i+1}/{len(joined_actions)}")
                    observation, reward, done, truncated, info = env.step(joined_actions[i])
                    if is_error_response(observation.processed_text) or truncated: 
                        print(f"Action {i} during redo failed: {observation.processed_text[:100]}")
                        attempts[i] = attempts[i] + 1
                        if attempts[i] >= MAX_ATTEMPTS:
                            joined_actions.remove(joined_actions[i])
                            # If action has failed multiple times, remove it from replay
                        continue
                successful_replay = True
            
            # After successful replay, get a fresh observation
            observation = env.get_obs()
            continue
        
        # Get judgment after this step
        current_judgment = judge(
            observations=processed_text_history,
            actions=[a.response for a in action_history],
            instruction=task
        )
        # Store judgment
        judgment_history.append(current_judgment)
        
        # Print judgment information
        if current_judgment and hasattr(current_judgment, 'values') and current_judgment.values:
            success_value = current_judgment.values.get('success', False)
            task_success = current_judgment.values.get('task_success', False)
            print(f"Step {step_num} Judgment - Success: {success_value}, Task Success: {task_success}")
            print(f"Judgment Response: {current_judgment.response[:10000]}...")

        # Check if task is complete according to judgment
        if current_judgment and hasattr(current_judgment, 'values') and current_judgment.values:
            if current_judgment.values.get('success') >= success_threshold or current_judgment.values.get('task_success'):
                print(f"Task completed successfully according to judge after {i} steps.")
                break
        
        # Check if done based on agent response
        if action.response and "DONE" in action.response:
            print(f"Agent signaled completion after {i} steps but judge disagreed.")
            observation.processed_text.append("\n\nNOTE: You signaled completion in the prior action by inputting DONE, but the judge disagreed. Do not put DONE as your action this time.")
            # Somewhat hacky way to tell the agent to not say "DONE" again
            action_history.pop()
            thoughts_history.pop()
        
        # If environment signals done, we're finished
        if done:
            print(f"Environment signaled completion after {i} steps.")
            break
    
    # Determine final success based on last judgment
    success = False
    if judgment_history:
        final_judgment = judgment_history[-1]
        if final_judgment and hasattr(final_judgment, 'values') and final_judgment.values:
            if final_judgment.values.get('success') >= success_threshold or final_judgment.values.get('task_success'):
                success = True
    
    print(f"Task {'completed successfully' if success else 'did not complete'} within {len(action_history)} steps.")
    return success, observation.processed_text, action_history, thoughts_history, judgment_history, env

def solve_task_insta_with_agent_traj(task, website, model='gpt-4o', max_steps=10, env=None, previous_actions: list[BrowserAction] | None = None, client_kwargs=None, **kwargs):
    """
    Solve a task using the InSTA pipeline's generate_trajectory function.
    
    Args:
        task (str): The task to solve
        website (str): The website URL
        model (str): The model to use
        max_steps (int): Maximum number of steps
        env (InstaEnv, optional): Existing environment to use. If None, creates a new one.
        previous_actions (list, optional): Previous successful actions to replay before starting new actions.
        client_kwargs (dict, optional): Additional kwargs for the client configuration.
        
    Returns:
        tuple: (success, final_observation, action_history, thoughts_history, env)
    """
    
    # Create environment if not provided
    if env is None:
        try:
            browser_config = get_browser_config(
                playwright_port=3000 + random.randint(0, 8),
            )
            env = InstaEnv(config=browser_config)
        except ConnectionRefusedError as e:
            print(f"Connection refused error: {e}")
            print("The InSTA browser environment server is not running or not accessible.")
            print("Please ensure the Docker container is running with ports 3000-3008 exposed.")
            return False, "Connection error: Browser environment not accessible", [], [], None
        except Exception as e:
            print(f"Error creating environment: {e}")
            return False, f"Environment error: {str(e)}", [], [], None
    
    # Use the appropriate client_kwargs based on the model
    if client_kwargs is None:
        if model.startswith("claude"):
            client_kwargs = {
                "api_key": secrets.get("ANTHROPIC_API_KEY"),
                "base_url": "https://api.anthropic.com/v1"
            }
        elif model.startswith("gemini"):
            client_kwargs = {
                "api_key": secrets.get("GEMINI_API_KEY"),
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
            }
        else:
            client_kwargs = {
                "api_key": secrets.get("OPENAI_API_KEY")
                # No base_url needed for OpenAI (it uses the default)
            }
    
    # Configure agent and judge
    agent_config = get_agent_config(
        client_kwargs=client_kwargs,
        generation_kwargs={"model": model},
        max_obs_tokens=4096,
        catch_errors=True,
        max_errors=5,
        log_errors=True
    )
    
    judge_config = get_judge_config(
        client_kwargs=client_kwargs,
        generation_kwargs={"model": model},
        max_obs_tokens=4096,
        catch_errors=True,
        max_errors=5,
        log_errors=True
    )
    
    # Ensure website has http:// prefix
    if not website.startswith("http"):
        website = "http://" + website
    
    # Initialize agent and judge
    from insta.agent import BrowserAgent
    from insta.judge import BrowserJudge
    
    agent = BrowserAgent(config=agent_config)
    judge = BrowserJudge(config=judge_config)
    
    # Reset environment with the website
    observation, info = env.reset(url=website)
    
    # Replay previous actions if provided
    if previous_actions and len(previous_actions) > 0:
        print(f"Replaying {len(previous_actions)} previous actions...")
        
        # Initialize agent with first observation
        agent.push_observation(
            observation=observation.processed_text,
            instruction=task,
            current_url=website
        )
        
        for i, prev_action in enumerate(previous_actions):
            print(f"Replaying action {i+1}/{len(previous_actions)}")
            print(f"Action: {prev_action}")
            
            # Push the action to the agent
            agent.push_action(prev_action['matched_response'])
            
            # Take step in environment with previous action
            action_obj = BrowserAction(**prev_action)  # Turn dict into BrowserAction
            new_func_calls = [FunctionCall(**call) if type(call) == dict else call for call in action_obj.function_calls]
            # Check that all calls are FunctionCalls and not dicts
            action_obj.function_calls = new_func_calls

            observation, reward, done, truncated, info = env.step(action=action_obj)
            
            # Push the new observation to the agent
            agent.push_observation(
                observation=observation.processed_text,
                instruction=task,
                current_url=info.get('current_url', website)
            )
            
            # # If environment signals done during replay, we're finished
            # if done:
            #     print(f"Environment signaled completion during replay after {i+1} steps.")
            #     return True, observation.processed_text, previous_actions[:i+1], previous_actions[:i+1], env
    
    try:
        # Generate trajectory using the pipeline function
        observations, actions, judgment = generate_trajectory(
            agent=agent if previous_actions else agent_config,
            judge=judge_config,
            env=env,
            url=website if not previous_actions else None,  # Don't reset if we've replayed actions
            instruction=task,
            max_actions=max_steps
        )
        
        # Extract action history and thoughts history
        action_history = [action for action in actions]
        thoughts_history = [action["response"] for action in actions]
        
        # If we had previous actions, prepend them to the histories
        # if previous_actions:
        #     action_history = previous_actions + action_history
        #     thoughts_history = previous_actions + thoughts_history
        
        # Determine success based on judgment
        success = False
        if judgment and isinstance(judgment, dict):
            success_value = judgment.get('success', 0.0)
            task_success = judgment.get('task_success', False)
            success = (success_value >= 0.9) or task_success
        judgment_history = [judgment]
        
        # Get the final observation text
        final_observation = observations[-1]["processed_text"] if observations else ""
        observations_text = [obs["processed_text"] for obs in observations]
        
        print(f"Task {'completed successfully' if success else 'did not complete'} within {len(action_history)} steps.")
        return success, observations_text, action_history, thoughts_history, judgment_history, env
        
    except Exception as e:
        print(f"Error in generate_trajectory: {e}")
        return False, f"Error: {str(e)}", [], [], {}, env

# --------------------------------------------------
# NEW: Functions to call Anthropic and Gemini models using custom API keys

def solve_task_insta_with_anthropic(task, website, max_steps=10, **kwargs):
    """
    Solves a task using the Anthropic model via the OpenAI SDK compatibility layer.
    It loads the ANTHROPIC_API_KEY from secrets.json, sets the API base to Anthropic's endpoint,
    and uses the 'claude-3.7-sonnet' model.
    """
    return solve_task_insta_with_agent(
        task, 
        website, 
        model="claude-3-7-sonnet-latest", 
        max_steps=max_steps, 
        client_kwargs={
            "api_key": secrets.get("ANTHROPIC_API_KEY"),
            "base_url": "https://api.anthropic.com/v1"
        }
    )


def solve_task_insta_with_gemini(task, website, max_steps=10, **kwargs):
    """
    Solves a task using the Gemini model via the OpenAI SDK compatibility layer.
    It loads the GEMINI_API_KEY from secrets.json, sets the API base to Gemini's endpoint,
    and uses the 'gemini-2.5-pro' model.
    """
    return solve_task_insta_with_agent(
        task, 
        website, 
        model="gemini-2.5-pro-preview-05-06", 
        max_steps=max_steps, 
        client_kwargs={
            "api_key": secrets.get("GEMINI_API_KEY"),
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
        }
    )

# --------------------------------------------------

def restore_from_checkpoints(tool, checkpoints):
    """
    Attempt to restore the session from checkpoints, starting with the most recent.
    
    Args:
        tool: The InSTA tool object.
        checkpoints: List of checkpoints to try.
        
    Returns:
        tuple or None: (outputs, session_id, current_checkpoint) if successful, None otherwise.
    """
    # Try checkpoints from most recent to oldest
    for checkpoint_idx in range(len(checkpoints) - 1, -1, -1):
        checkpoint = checkpoints[checkpoint_idx]
        print(f"Trying checkpoint {checkpoint_idx+1}: URL={checkpoint['url']}, with {len(checkpoint['actions'])} actions")
        
        # First navigate to the checkpoint URL
        checkpoint_outputs = tool(url=checkpoint["url"])
        
        # Check if navigation was successful
        if is_error_response(checkpoint_outputs):
            print(f"Failed to navigate to checkpoint URL: {checkpoint['url']}")
            continue
        
        # Get the new session ID
        session_id = extract_session_id(checkpoint_outputs)
        if not session_id:
            print("No session ID found after checkpoint navigation")
            continue
            
        print(f"New session ID after checkpoint navigation: {session_id}")
        
        # Try to replay actions from this checkpoint
        replay_success = True
        replay_outputs = checkpoint_outputs
        
        for action_idx, action in enumerate(checkpoint["actions"]):
            print(f"Replaying action {action_idx+1}/{len(checkpoint['actions'])}: {action[:50]}...")
            action_outputs = tool(session_id=session_id, action=action)
            
            # Check for errors during replay
            if is_error_response(action_outputs):
                print(f"Failed to replay action {action_idx+1} from checkpoint {checkpoint_idx+1}")
                replay_success = False
                break
            
            # Update session ID if it changed
            new_session_id = extract_session_id(action_outputs)
            if new_session_id and new_session_id != session_id:
                print(f"Session ID changed during replay from {session_id} to {new_session_id}")
                session_id = new_session_id
            
            replay_outputs = action_outputs
        
        if replay_success:
            print(f"Successfully restored using checkpoint {checkpoint_idx+1}")
            # Return the current state and checkpoint
            current_checkpoint = {
                "url": checkpoint["url"],
                "actions": checkpoint["actions"].copy()  # Copy the actions
            }
            return replay_outputs, session_id, current_checkpoint
    
    # If we get here, all checkpoints failed
    return None

def sample_task(tool, dataset, artificial_init_task=False, max_retries=10):
    """Sample a task from the dataset and ensure the website loads properly."""
    retry_count = 0
    while retry_count < max_retries:
        sample = dataset[random.randint(0, len(dataset)-1)]
        website = sample.get("domain", "unknown website")
        task = sample.get("task", "unknown task")
        assert website != "unknown website"
        assert task != "unknown task"
        print(f"Attempting to reach website: {website}")
        
        try:
            outputs = tool.reset(url="http://" + website)
            
            # Check for environment errors
            if is_error_response(outputs[0].processed_text):
                print(f"Environment error detected for {website}: {outputs}")
                retry_count += 1
                time.sleep(10)
                continue
                
            # Extract webpage text
            # webpage_text = extract_webpage_text(outputs)
            webpage_text = outputs[0].processed_text[:]
                
            # Check if webpage text is empty or too short
            if len(webpage_text) < 50:  # Arbitrary minimum length for a meaningful webpage
                print(f"Website {website} loaded but content is too minimal. Retrying...")
                print(webpage_text)
                retry_count += 1
                continue
                
        except Exception as e:
            print(f"Error accessing website {website}: {e}")
            retry_count += 1
            continue
            
        # If we get here, the website loaded successfully
        if artificial_init_task:
            print("Generating artificial initial task...")
            try:
                task = initial_task_propose(website, webpage_text)
            except Exception as e:
                print(f"Error generating initial task: {e}")
                retry_count += 1
                continue
            
        return website, task, webpage_text
    
    # If we've exhausted all retries
    raise Exception(f"Failed to find a working website after {max_retries} attempts")

def run_task_generator(dataset, artificial_init_task=False, num_tasks=10, num_steps=10, model='gpt-4o-mini', use_browser_agent=False, use_gen_traj=True):
    """Run the task generator to create a sequence of tasks and solve them."""
    tool = InstaEnv()
    out = []
    
    # Try to find a working website
    max_website_attempts = 5
    for attempt in range(max_website_attempts):
        try:
            website, overall_task, webpage_text = sample_task(tool, dataset, artificial_init_task)
            print(f"Website: {website}\nTask: {overall_task}")
            print(f"Webpage text length: {len(webpage_text)} characters")
            break
        except Exception as e:
            print(f"Error finding a working website (attempt {attempt+1}/{max_website_attempts}): {e}")
            if attempt == max_website_attempts - 1:
                print("Failed to find any working website after multiple attempts. Exiting.")
                return []
            time.sleep(10)  # Wait before retrying
    
    task_history = []
    outputs = None
    env = None  # Store environment for replay
    successful_actions = []  # Store successful actions for replay
    
    if not use_browser_agent:
        outputs = tool(url="http://" + website)  # Initial page load
        
        # Check for environment errors
        if is_error_response(outputs):
            print(f"Environment error detected: {outputs}")
            print("Cannot proceed with task generation due to environment error.")
            return []
            
        print(f"Initial page load response length: {len(outputs)} characters")
        session_id = extract_session_id(outputs)
        
        if session_id:
            print(f"Initial session ID: {session_id}")
        else:
            print("WARNING: No session ID found in initial page load!")
    
    task = overall_task if artificial_init_task else followup_task_propose(
        website, webpage_text, task_history, model=model, overall_task=overall_task)
    print("Initial Task:", task)
    time.sleep(1)
    
    for task_idx in range(num_tasks):
        print(f"\n--- Starting task {task_idx+1}/{num_tasks}: {task} ---\n")
        
        # Check if environment is still working (only for non-BrowserAgent)
        if not use_browser_agent and is_error_response(outputs):
            print("Environment error detected. Attempting to restart...")
            outputs = tool(url="http://" + website)
            if is_error_response(outputs):
                print("Environment still not working. Skipping this task sequence.")
                break
        
        if use_browser_agent:
            # Use the BrowserAgent approach with replay
            # Replay should only occur when previous task was not completed
            method = solve_task_insta_with_agent_traj if use_gen_traj else solve_task_insta_with_agent
            completed, webpage_text, action_history, thoughts_history, judgment_history, env = method(
                task, website, model=model, max_steps=num_steps, env=env, 
                previous_actions=successful_actions if task_idx > 0 and not completed else None,
                client_kwargs=None,
            )
        else:
            # Use the original approach
            completed, webpage_text, action_history, thoughts_history = solve_task_insta(
                tool, task, website, outputs, model=model, max_steps=num_steps)
            judgment_history = None  # No judgment for the original approach
        
        if completed:
            # Task completed successfully
            task_history.append(task)
            print(f"Successfully completed task: {task}")
            
            # Store successful actions for replay
            if use_browser_agent:
                # Only store actions up to the point where the task was completed
                # This prevents storing unnecessary actions after task completion
                successful_actions.extend(action_history)
                
                # If we have judgments, print final judgment
                if judgment_history and len(judgment_history) > 0:
                    final_judgment = judgment_history[-1]
                    final_judgment = final_judgment["response"] if type(final_judgment) == dict else final_judgment.response
                    print(f"Final Judgment: {final_judgment}")
            
            # Generate followup task
            try:
                followup = followup_task_propose(website, webpage_text, task_history, 
                                               model=model, overall_task=overall_task)
                print("Followup Task:", followup)
                task = followup
                
                # Include judgment history in the output if available
                # if judgment_history:
                #     out.append((website, task, action_history, thoughts_history, webpage_text, judgment_history))
                # else:
                #     out.append((website, task, action_history, thoughts_history, webpage_text))
                file_action_history = [
                    (action['matched_response'] if use_gen_traj else action.matched_response) for action in action_history
                ] if use_browser_agent else action_history
                out.append((website, task, file_action_history, thoughts_history, webpage_text))
            except Exception as e:
                print(f"Error generating followup task: {e}")
                # Try a new website if we can't generate a followup
                try:
                    website, overall_task, webpage_text = sample_task(tool, dataset, artificial_init_task)
                    print(f"New Website: {website}\nTask: {overall_task}")
                    task = overall_task if artificial_init_task else followup_task_propose(
                        website, webpage_text, [], model=model, overall_task=overall_task)
                    print("Initial Task:", task)
                    
                    if use_browser_agent:
                        # Reset environment and successful actions for new website
                        env = None
                        successful_actions = []
                    else:
                        outputs = tool(url="http://" + website)
                        
                    task_history = []
                except Exception as e2:
                    print(f"Error finding a new website: {e2}")
                    break
        else:
            # Task failed to complete
            print(f"Task \"{task}\" did not complete within the maximum steps.")
            
            if len(task_history) == 0:
                # If no successful tasks yet, try a new website
                try:
                    website, overall_task, webpage_text = sample_task(tool, dataset, artificial_init_task)
                    print(f"New Website: {website}\nTask: {overall_task}")
                    task = overall_task if artificial_init_task else followup_task_propose(
                        website, webpage_text, [], model=model, overall_task=overall_task)
                    print("Initial Task:", task)
                    
                    if use_browser_agent:
                        # Reset environment and successful actions for new website
                        # Replay handled by solve_task_insta_with_agent
                        env = None
                        successful_actions = []
                    else:
                        outputs = tool(url="http://" + website)
                except Exception as e:
                    print(f"Error finding a new website: {e}")
                    break
            else:
                # Try to generate a new task based on the current state
                try:
                    task = followup_task_propose(website, webpage_text, task_history, 
                                               model=model, overall_task=overall_task)
                    print("New Task after failed attempt:", task)
                except Exception as e:
                    print(f"Error generating new task after failed attempt: {e}")
                    # Try a new website
                    try:
                        website, overall_task, webpage_text = sample_task(tool, dataset, artificial_init_task)
                        print(f"New Website: {website}\nTask: {overall_task}")
                        task = overall_task if artificial_init_task else followup_task_propose(
                            website, webpage_text, [], model=model, overall_task=overall_task)
                        print("Initial Task:", task)
                        
                        if use_browser_agent:
                            # Reset environment and successful actions for new website
                            env = None
                            successful_actions = []
                        else:
                            outputs = tool(url="http://" + website)
                            
                        task_history = []
                    except Exception as e2:
                        print(f"Error finding a new website: {e2}")
                        break
    
    return out

def summarize_task_sequence(task_sequence, model='o3-mini', max_retries=3):
    """
    Summarize a sequence of tasks into a single task with a sequence of actions.
    
    Args:
        task_sequence: A sequence of tasks to summarize.
        model (str, optional): The model to use for summarization. Defaults to 'o3-mini'.
        max_retries (int, optional): Maximum number of retries. Defaults to 3.
        
    Returns:
        dict: A dictionary containing the summarized task, website, action sequence, and thoughts sequence.
              Returns None if summarization fails.
    """
    sys_prompt = SYS_TASK_SUMMARIZE
    user_prompt = f"Task sequence: {task_sequence}"
    
    for attempt in range(max_retries):
        try:
            if model.startswith('gpt'):
                output = call_gpt_4o(sys_prompt, user_prompt, model=model)
            else:
                output = call_o3(sys_prompt, user_prompt, model=model)
            
            if output == "FAILED":
                return None
                
            data = parse_json(output)
            if data:
                return data
        except Exception as e:
            print(f"Error in summarizing task sequence (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(1)
    
    print(f"Failed to summarize task sequence after {max_retries} attempts")
    return None

def summarize_existing_task_sequences(input_file='task_sequences.jsonl', output_file='summarized_from_existing.jsonl', model='gpt-4o-mini', max_retries=3):
    """
    Summarize existing task sequences from a JSONL file.
    
    Args:
        input_file (str, optional): Path to the input JSONL file containing task sequences. Defaults to 'task_sequences.jsonl'.
        output_file (str, optional): Path to the output JSONL file for summarized tasks. Defaults to 'summarized_from_existing.jsonl'.
        model (str, optional): The model to use for summarization. Defaults to 'gpt-4o-mini'.
        max_retries (int, optional): Maximum number of retries for summarization. Defaults to 3.
        
    Returns:
        int: Number of successfully summarized task sequences.
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return 0
        
    # Read task sequences from file
    task_sequences = []
    try:
        with open(input_file, 'r') as f:
            for line in f:
                try:
                    task_data = json.loads(line.strip())
                    task_sequences.append(task_data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {e}")
    except Exception as e:
        print(f"Error reading from {input_file}: {e}")
        return 0
        
    print(f"Loaded {len(task_sequences)} task sequences from {input_file}")
    
    # Group tasks by website
    tasks_by_website = {}
    for task in task_sequences:
        website = task.get('website')
        if not website:
            continue
        if website not in tasks_by_website:
            tasks_by_website[website] = []
        tasks_by_website[website].append(task)
    
    print(f"Grouped tasks into {len(tasks_by_website)} websites")
    
    # Simple system prompt for task summarization only
    sys_prompt = """
    You are a helpful assistant that summarizes a sequence of tasks.
    Given a list of tasks that a user performed on a website, summarize them into a single comprehensive task.
    The summary should capture the overall goal of what the user was trying to accomplish across all the tasks.
    Return ONLY the summarized task as a single string, with no additional text or explanation.
    """
    
    # Summarize tasks for each website
    summarized_count = 0
    with open(output_file, 'a') as f:
        for website, website_tasks in tqdm(tasks_by_website.items(), desc="Summarizing websites"):
            try:
                # Only process websites with multiple tasks
                if len(website_tasks) <= 1:
                    continue
                
                # Extract just the task descriptions
                task_descriptions = [task.get('task', '') for task in website_tasks if task.get('task')]
                if not task_descriptions:
                    continue
                
                # Combine all action sequences
                all_actions = []
                for task in website_tasks:
                    actions = task.get('action_sequence', [])
                    if actions and isinstance(actions, list):
                        for action in actions:
                            # Handle different action formats
                            if action == "DONE":
                                continue  # Skip DONE actions except for the last one
                            
                            # If action is a string that contains JSON, parse it
                            if isinstance(action, str) and (action.startswith('{') or action.startswith('[')):
                                try:
                                    parsed_action = json.loads(action)
                                    all_actions.append(parsed_action)
                                except json.JSONDecodeError:
                                    all_actions.append(action)  # Keep as string if parsing fails
                            else:
                                all_actions.append(action)  # Keep as is (dict or string)
                
                # Add a final DONE action if needed
                if all_actions and all_actions[-1] != "DONE":
                    all_actions.append("DONE")
                
                # Combine all thought sequences
                all_thoughts = []
                for task in website_tasks:
                    thoughts = task.get('thoughts_sequence', [])
                    if thoughts and isinstance(thoughts, list):
                        all_thoughts.extend(thoughts)
                
                # Summarize only the tasks
                user_prompt = f"Here are the tasks performed on website {website}:\n" + "\n".join([f"- {task}" for task in task_descriptions])
                
                # Try to get a summarized task
                summarized_task = None
                for attempt in range(max_retries):
                    try:
                        if model.startswith('gpt'):
                            llm_output = call_gpt_4o(sys_prompt, user_prompt, model=model)
                        else:
                            llm_output = call_o3(sys_prompt, user_prompt, model=model)
                        
                        # Clean up the output - we just want the raw summarized task
                        summarized_task = llm_output.strip()
                        if summarized_task:
                            break
                    except Exception as e:
                        print(f"Error in task summarization (attempt {attempt+1}/{max_retries}): {e}")
                        time.sleep(1)
                
                if not summarized_task:
                    continue
                
                # Create the summarized entry
                summary = {
                    'website': website,
                    'task': summarized_task,
                    'action_sequence': all_actions,
                    'thoughts_sequence': all_thoughts
                }
                
                f.write(json.dumps(summary) + "\n")
                summarized_count += 1
                
            except Exception as e:
                print(f"Error summarizing sequence for {website}: {e}")
    
    print(f"Successfully summarized {summarized_count} task sequences to {output_file}")
    return summarized_count

def ensure_container_running():
    """Check if the required Docker container is running and start it if needed."""
    import subprocess
    import time
    
    # Check if container is already running
    result = subprocess.run(
        ["sudo", "docker", "ps", "--filter", "publish=3000", "--format", "{{.ID}}"],
        capture_output=True, text=True
    )
    
    if result.stdout.strip():
        print("Container already running")
        return True
        
    print("Starting container...")
    subprocess.Popen(
        ["sudo", "docker", "run", "-p", "7860:7860", "-p", "3000-3007:3000-3007", 
         "-t", "brandontrabucco/insta-browser-environment"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    # Wait for container to start
    for _ in range(10):
        time.sleep(2)
        result = subprocess.run(
            ["sudo", "docker", "ps", "--filter", "publish=3000", "--format", "{{.ID}}"],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            print("Container started successfully")
            # Give services a moment to initialize
            time.sleep(5)
            return True
            
    print("Failed to start container")
    return False

# --- Main execution ---
def main():
    """
    Main function to parse arguments and run the appropriate environment.
    
    This function:
    1. Parses command line arguments to determine which environment to run
    2. For InSTA: Generates task sequences, writes them to files, and summarizes them
    3. For OSWorld: Generates and solves tasks in the local desktop environment
    4. For summarize: Summarizes existing task sequences from a file
    """
    parser = argparse.ArgumentParser(description="Generate Task Traces for InSTA and OSWorld Environments")
    parser.add_argument("--env", choices=["insta", "osworld", "summarize"], required=True,
                        help="Environment to run: 'insta' for InSTA (text-based), 'osworld' for OSWorld (image-based), 'summarize' for summarizing existing sequences")
    parser.add_argument("--num_task_seq", type=int, default=5,
                        help="Number of task sequences to generate")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers to use for task generation")
    parser.add_argument("--num_task_per_seq", type=int, default=10,
                        help="Number of tasks per sequence")
    parser.add_argument("--num_steps_per_task", type=int, default=10,
                        help="Number of steps per task")
    parser.add_argument("--input_file", type=str, default="task_sequences.jsonl",
                        help="Input file for summarization (only used with --env=summarize)")
    parser.add_argument("--output_file", type=str, default="summarized_task_sequences.jsonl",
                        help="Output file for summarization (only used with --env=summarize)")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini",
                        help="Model to use for summarization (only used with --env=summarize)")
    parser.add_argument("--write_to_logs", action="store_true",
                        help="Write to logs (only used with --env=insta)")
    parser.add_argument("--log_file", type=str, default=f"logs/logs_{time.time()}.txt",
                        help="Log file for task generation (only used with --env=insta)")
    args = parser.parse_args()

    if args.env == "insta":
        # if not setup_runpod_endpoint():
        #     print("Failed to connect to RunPod endpoint. Exiting.")
        #     return
        
        ensure_container_running()
        dataset = load_dataset("data-for-agents/insta-150k", split="train[:1000]")
        print(f"Successfully loaded dataset with {len(dataset)} items")
        if args.write_to_logs:
            sys.stdout = open(args.log_file, 'w')
            sys.stderr = open(args.log_file, 'w')
            # Force UTF-8 encoding for stdout/stderr
            # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            # sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
            
        task_sequences = []
        try:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(
                    run_task_generator, 
                    dataset, 
                    True, 
                    args.num_task_per_seq, 
                    args.num_steps_per_task, 
                    args.model,
                    use_browser_agent=True
                ) for _ in range(args.num_task_seq)]
                for future in as_completed(futures):
                    result = future.result()
                    if result:  # Check if result is not empty
                        task_sequences.append(result)
                    # try:
                    #     result = future.result()
                    #     if result:  # Check if result is not empty
                    #         task_sequences.append(result)
                    # except Exception as e:
                    #     print(f"Error in task sequence generation: {e}")
        except Exception as e:
            print(f"Error in thread pool execution: {e}")
            
        print(f"Processing {len(task_sequences)} task sequences")
        # Flatten task sequences with error handling
        flat_task_sequences = []
        for sublist in task_sequences:
            if not sublist:
                continue
            for item in sublist:
                try:
                    # Validate item format
                    if not isinstance(item, tuple) or len(item) != 5:
                        print(f"Skipping malformed item: {item}")
                        continue  
                    website, task, action_sequence, thoughts_sequence, webpage_text = item
                    # Basic validation of fields
                    if not isinstance(website, str) or not website:
                        print(f"Invalid website in item: {item}")
                        continue
                    if not isinstance(task, str) or not task:
                        print(f"Invalid task in item: {item}")
                        continue   
                    flat_task_sequences.append(item)
                except Exception as e:
                    print(f"Error processing task sequence item: {e}")
        
        print(f"Flattened to {len(flat_task_sequences)} individual tasks")
        # Write individual tasks to file
        if flat_task_sequences:
            try:
                print(f"Writing {len(flat_task_sequences)} individual tasks to task_sequences.jsonl")
                formatted_tasks_by_website = {}  # Store the formatted dictionaries
                
                with open('task_sequences.jsonl', 'a') as f:
                    for seq in flat_task_sequences:
                        try:
                            website, task, action_sequence, thoughts_sequence, webpage_text = seq
                            # Ensure serializable data
                            task_data = {
                                "website": str(website),
                                "task": str(task),
                                "action_sequence": action_sequence,
                                "thoughts_sequence": thoughts_sequence,
                                "webpage_text": str(webpage_text)
                            }
                            
                            f.write(json.dumps(task_data) + "\n")
                            
                            # Group by website for summarization
                            if website not in formatted_tasks_by_website:
                                formatted_tasks_by_website[website] = []
                            formatted_tasks_by_website[website].append(task_data)
                            
                        except Exception as e:
                            print(f"Error writing task to file: {e}")
            except Exception as e:
                print(f"Error writing to task_sequences.jsonl: {e}")
                
            # Summarize tasks for each website
            summarized_task_sequences = []
            for website, formatted_sequences in formatted_tasks_by_website.items():
                try:
                    summary = summarize_task_sequence(formatted_sequences, model="gpt-4.1-mini")
                    if summary:
                        summarized_task_sequences.append(summary)
                except Exception as e:
                    print(f"Error summarizing sequence for {website}: {e}")
            
            # Write summarized tasks to file
            if summarized_task_sequences:
                try:
                    print(f"Writing {len(summarized_task_sequences)} summarized tasks to summarized_task_sequences.jsonl")
                    with open('summarized_task_sequences.jsonl', 'a') as f:
                        for summary in summarized_task_sequences:
                            try:
                                filtered = {k: summary[k] for k in ['task', 'website', 'action_sequence', 'thoughts_sequence'] if k in summary}
                                if filtered:
                                    f.write(json.dumps(filtered) + "\n")
                            except Exception as e:
                                print(f"Error writing summary to file: {e}")
                except Exception as e:
                    print(f"Error writing to summarized_task_sequences.jsonl: {e}")
            
            print("InSTA task generation and summarization completed")
        else:
            print("No valid task sequences were generated")
            
    elif args.env == "osworld":
        possible_local_software = ['VS Code', 'Chrome', 'ThunderBird Mail', 'LibreOffice Writer',
                                   'LibreOffice Calc', 'LibreOffice Impress', 'Files',
                                   'Calendar', 'AisleRiot Solitare']
        software = random.choice(possible_local_software)
        try:
            env = DesktopEnv(
                provider_name="vmware",
                path_to_vm="C:\\Users\\dylan\\Synth-Dataset\\vmware_vm_data\\Ubuntu0\\Ubuntu0.vmx",
                snapshot_name="init_state",
                action_space="pyautogui",
                require_a11y_tree=True
            )
            obs, reward, done, info = env.step("pyautogui.rightClick()")
        except Exception as e:
            print(f"Error setting up DesktopEnv: {e}")
            return
        base64_image = encode_image_from_variable(obs['screenshot'])
        a11y_tree = obs['accessibility_tree']
        task = initial_task_propose_local(software, base64_image)
        print("Initial Task:", task)
        trajectories = []
        task_history = []
        num_iterations = 1
        completed, thoughts_history, command_history = solve_task_osworld(task, env, base64_image, a11y_tree, time_btw_tasks=1, max_steps=15)
        for i in range(num_iterations):
            if completed:
                trajectories.append((software, task, command_history, thoughts_history))
                task_history.append(task)
                task = followup_task_propose_local(software, task_history, base64_image)
                print("Followup Task:", task)
                null_action = "pyautogui.moveTo(960, 540); time.sleep(0.5)"
                obs, reward, done, info = env.step(null_action)
                base64_image = encode_image_from_variable(obs['screenshot'])
                a11y_tree = obs['accessibility_tree']
                completed, thoughts_history, command_history = solve_task_osworld(task, env, base64_image, a11y_tree, time_btw_tasks=1, max_steps=15)
            else:
                completed, thoughts_history, command_history = solve_task_osworld(task, env, base64_image, a11y_tree, time_btw_tasks=1, max_steps=15)
            print(f"Run {i+1} of {num_iterations} completed: {completed}")
        if completed:
            trajectories.append((software, task, command_history, thoughts_history))
            task_history.append(task)
        with open('trajectories.jsonl', 'a') as f:
            for traj in trajectories:
                f.write(json.dumps({
                    "software": traj[0],
                    "task": traj[1],
                    "action_history": traj[2],
                    "thoughts_history": traj[3]
                }) + "\n")
        print("OSWorld task generation completed. Trajectories saved to trajectories.jsonl.")
    elif args.env == "summarize":
        print(f"Summarizing task sequences from {args.input_file} to {args.output_file}")
        summarize_existing_task_sequences(
            input_file=args.input_file,
            output_file=args.output_file,
            model=args.model
        )
        print("Summarization completed")

if __name__ == "__main__":
    main()
