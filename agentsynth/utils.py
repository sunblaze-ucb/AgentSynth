#%%
from prompts import SYS_TASK_ACTION, SYS_COMPUTER_ACTION, SYS_TASK_SUMMARY, SYS_SUBTASK_SUMMARY, SYS_VERIFIER, SYS_VERIFIER_KEY_INFO, SYS_VERIFIER_KEY_SCREEN, SYS_VERIFIER_VERDICT, SYS_TASK_INIT_PERSONA, SYS_TASK_FOLLOWUP_PERSONA, SYS_INFO_SUMMARY
import requests
import json
import re
import base64
from PIL import Image
from io import BytesIO
from parse_computer_use import parse_computer_use_pyautogui
import random
import time
from datetime import datetime
import openai
import os

#%%

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }

def call_llms(sys_prompt, user_prompt, img, model = 'gpt-4.1'):
    if type(img) == list:
        screenshots_payload = [{"type": "image_url", "image_url": f"data:image/png;base64,{item}"} for item in img]
    else:
        screenshots_payload = [{"type": "image_url", "image_url": f"data:image/png;base64,{img}"}]

    client = openai.OpenAI(
    api_key = OPENAI_API_KEY,
    )

    response = client.chat.completions.create(
        model = model, 
        temperature = 1.0, 
        # truncation = 'auto',
        messages = [
            {
                "role": "system", 
                "content": [
                    {
                        "type": "text", 
                        "text": sys_prompt
                        }
                ]
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text", 
                        "text": user_prompt
                    },
                    *screenshots_payload
                ]
            }
        ]
    )

    # if response.status != 'completed':
    #     print(f"Error: {response.status}")
    #     raise Exception("API call failed")
    current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('F:/docker/token_count_llm.txt', 'a') as f:
        f.write(f"gpt: {current_date_time} {model} {response.usage.total_tokens}\n")
    # print(response.json())
    return response.choices[0].message.content



def call_gpt(sys_prompt, user_prompt, img, model = 'gpt-4.1'):
    if type(img) == list:
        screenshots_payload = [{"type": "input_image", "image_url": f"data:image/png;base64,{item}"} for item in img]
    else:
        screenshots_payload = [{"type": "input_image", "image_url": f"data:image/png;base64,{img}"}]
    payload = {
        "model": model, 
        "temperature": 1.0, 
        'truncation': 'auto',
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
                    *screenshots_payload
                ]
            }
        ]
    }
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=payload
    )
    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        raise Exception("API call failed")
    current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('F:/docker/token_count_gpt.txt', 'a') as f:
        f.write(f"gpt: {current_date_time} {model} {response.json()['usage']['total_tokens']}\n")
    # print(response.json())
    return response.json()['output'][0]['content'][0]['text']






def call_computer_use_preview(sys_prompt, user_prompt, img, model = "computer-use-preview"):
    if type(img) == list:
        screenshots_payload = [{"type": "input_image", "image_url": f"data:image/png;base64,{item}"} for item in img]
    else:
        screenshots_payload = [{"type": "input_image", "image_url": f"data:image/png;base64,{img}"}]

    payload = {
        "model": model, 
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
                    *screenshots_payload
                ]
            }
        ]
    }
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=payload
    )
    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        raise Exception("API call failed")
    current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('F:/docker/token_count_computer.txt', 'a') as f:
        f.write(f"computer: {current_date_time} {model} {response.json()['usage']['total_tokens']}\n")
    # print('computer use total tokens: ', response.json()['usage']['total_tokens'])
    return response.json()


def parse_computer_use_preview(response):
    raw_output = response['output']
    for item in raw_output:
        if item['type'] == 'computer_call':
            action = item['action']
            return parse_computer_use_pyautogui(action)
    print("No computer call found in the response.")
    return None

#%%
def parse_json(llm_output):
    match = re.search(r"```json\s*(\{.*?\})\s*```", llm_output, re.DOTALL)
    match_2 = re.search(r"\{.*?\}", llm_output, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
            print("Extracted JSON:", data)
            return data
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e, llm_output)
            return None
    elif match_2:
        json_str = match_2.group(0)
        try:
            data = json.loads(json_str)
            print("Extracted JSON:", data)
            return data
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e, llm_output)
            return None
    else:
        print("No JSON block found.", llm_output)
        return None
    


def initial_task_propose_persona(persona, img):
    sys_prompt = SYS_TASK_INIT_PERSONA
    user_prompt = f"You are {persona}, what task would you perform on the computer?"
    err_count = 0
    while True:
        try:
            llm_output = call_gpt(sys_prompt, user_prompt, img)
            task_info = parse_json(llm_output)['task']
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(10)

            # err_count += 1
            # if err_count > 10:
            #     print("Error count exceeded limit. Exiting...")
            #     return None
            
            continue

    return task_info


def followup_task_propose_persona(persona, task_history, img, failed_task = None):
    sys_prompt = SYS_TASK_FOLLOWUP_PERSONA
    user_prompt = f"You are {persona}. Given the task history {task_history}, what would be a followup task?"
    if failed_task:
        user_prompt += f" Note that these tasks {failed_task} are too hard for the agent, propose a simplier one."

    err_count = 0
    while True:
        try:
            llm_output = call_gpt(sys_prompt, user_prompt, img)
            task_info = parse_json(llm_output)['task']
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(10)

            # err_count += 1
            # if err_count > 10:
            #     print("Error count exceeded limit. Exiting...")
            #     return None
            
            continue
    return task_info


def generate_action(task, thoughts_history, action_history, info_history, img):
    sys_prompt = SYS_TASK_ACTION
    user_prompt = f"Given the task: {task}. You have gathered some information {info_history}. Here is your previous thinking process to complete the task {thoughts_history}. Here is your previous actions tried {action_history}. Here is the current screenshot, what would be the next action?"
    while True:
        try:
            llm_output = call_gpt(sys_prompt, user_prompt, img)
            parsed_output = parse_json(llm_output)
            action_info = parsed_output['action']
            thoughts = parsed_output['thoughts']
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(10)
            continue

    return action_info, thoughts

def generate_computer_use_action(task, step, command_history, img):
    sys_prompt = SYS_COMPUTER_ACTION
    user_prompt = f"Given the task: {task}, you have done the following actions: {command_history}. Next, you need to do the next step: {step}. What would be the action?"
    while True:
        try:
            llm_output = call_computer_use_preview(sys_prompt, user_prompt, img)
            action_info = parse_computer_use_preview(llm_output)
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(10)
            continue

    return action_info

def generate_key_info(task, thoughts, img):
    sys_prompt = SYS_INFO_SUMMARY
    user_prompt = f"Given the task: {task}. Here is your previous thinking process to complete the task {thoughts}. Here is the previous screenshots, what would be the key information summarized from these thoughts and actions?"
    while True:
        try:
            llm_output = call_gpt(sys_prompt, user_prompt, img)
            key_info = parse_json(llm_output)['info']
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(10)
            continue
    
    return key_info

def generate_summary(task_history, img):
    sys_prompt = SYS_TASK_SUMMARY
    user_prompt = f"Given the subtasks history {task_history} and the final screenshot, what would be a single task description that will be accomplished by performing these subtasks in the given sequence?"
    while True:
        try:
            llm_output = call_llms(sys_prompt, user_prompt, img)
            task_info = parse_json(llm_output)['task']
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(10)
            continue

    return task_info

def generate_subtask_summary(img):
    sys_prompt = SYS_SUBTASK_SUMMARY
    user_prompt = f"Given the set of screenshots of actions, what would be a single task description that will be accomplished by performing these actions in the given sequence?"
    while True:
        try:
            llm_output = call_gpt(sys_prompt, user_prompt, img)
            task_info = parse_json(llm_output)['task']
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(10)
            continue

    return task_info

def generate_verifier(task, screenshot_history, model = 'gpt-4.1'):
    sys_prompt = SYS_VERIFIER
    user_prompt = f"Given the task {task}, and the screenshot history, is the agent successful?"
    while True:
        try:
            llm_output = call_gpt(sys_prompt, user_prompt, screenshot_history, model)
            verifier_info = parse_json(llm_output)
            thoughts = verifier_info['thoughts']
            success_rate = verifier_info['success rate']
            success = verifier_info['success']
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(10)
            continue

    return success_rate, success, thoughts


def generate_verifier_key_points(task, model = 'gpt-4.1'):
    sys_prompt = SYS_VERIFIER_KEY_INFO
    user_prompt = f"Given the task {task}, what are the key points?"
    while True:
        try:
            llm_output = call_gpt(sys_prompt, user_prompt, [], model)
            verifier_info = parse_json(llm_output)
            thoughts = verifier_info['thoughts']
            key_points = verifier_info['key_points']
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(10)
            continue

    return key_points, thoughts

def generate_verifier_key_screen(task, key_points, img, model = 'gpt-4.1'):
    sys_prompt = SYS_VERIFIER_KEY_SCREEN
    user_prompt = f"Given the task {task}, the key points to finish the task {key_points}, and the screenshot of an action, is this screenshot a necessary step to complete the task?"
    while True:
        try:
            llm_output = call_gpt(sys_prompt, user_prompt, img, model)
            verifier_info = parse_json(llm_output)
            thoughts = verifier_info['thoughts']
            necessary = verifier_info['necessary']
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(10)
            continue

    return necessary, thoughts

def generate_verifier_verdict(task, key_points, img, model = 'gpt-4.1'):
    sys_prompt = SYS_VERIFIER_VERDICT
    user_prompt = f"Given the task {task}, the key points to finish the task {key_points}, and the screenshot of an action, is this screenshot a necessary step to complete the task?"
    while True:
        try:
            llm_output = call_gpt(sys_prompt, user_prompt, img, model)
            verifier_info = parse_json(llm_output)
            thoughts = verifier_info['thoughts']
            necessary = verifier_info['necessary']
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(10)
            continue

    return necessary, thoughts



def encode_image_from_variable(image_content):
    return base64.b64encode(image_content).decode('utf-8')

def decode_image_from_variable(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


def select_persona():
    with open('persona.jsonl', 'r') as file:
        data = file.readlines()
    selected_persona = random.choice(data)
    selected_persona = json.loads(selected_persona)
    return selected_persona['persona']


def resize_b64_images(data):
    img_data = base64.b64decode(data)
    img = Image.open(BytesIO(img_data))
    
    new_size = (img.width // 2, img.height // 2)
    resized_img = img.resize(new_size)
    
    buffered = BytesIO()
    resized_img.save(buffered, format="PNG")
    resized_b64 = base64.b64encode(buffered.getvalue()).decode()
            
    return resized_b64

# %%
