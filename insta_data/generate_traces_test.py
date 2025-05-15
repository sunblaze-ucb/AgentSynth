#%%
import requests
import json
import time
import re
from tqdm import tqdm
import random

persona = "A high school math teacher is teaching students the concepts of linear functions and definite integrals, helping them understand the relationships between functions and the methods for calculating the area of regions enclosed by curves."

software = "Chrome"

if os.path.exists('secrets.json'):
    with open('secrets.json', 'r') as f:
        secrets = json.load(f)
    OPENAI_API_KEY = secrets['OPENAI_API_KEY']
else:
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }

task_history = []
action_history = []

import sys
import os
if os.path.exists('E:\\Desktop2025.1.17\\CS 294 LLM Agent\\osworld\\OSWorld'):
    sys.path.append('E:\\Desktop2025.1.17\\CS 294 LLM Agent\\osworld\\OSWorld')
    sys.path.append('E:\\Desktop2025.1.17\\CS 294 LLM Agent')
    os.chdir('E:\\Desktop2025.1.17\\CS 294 LLM Agent\\osworld\\OSWorld')
else:
    home_dir = os.path.expanduser('~')
    sys.path.append(f'{home_dir}/Synth-Dataset')
    sys.path.append(f'{home_dir}/Synth-Dataset/OSWorld')
    os.chdir(f'{home_dir}/Synth-Dataset/OSWorld')
from parse_computer_use import parse_computer_use_pyautogui
from desktop_env.desktop_env import DesktopEnv
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

example = {
    "id": "94d95f96-9699-4208-98ba-3c3119edf9c2",
    "instruction": " ",
    "config": [
        {
            "type": "execute",
            "parameters": {
                "command": [
                    "python",
                    "-c",
                    "import pyautogui; import time; pyautogui.click(960, 540); time.sleep(0.5);"
                ]
            }
        }
    ],
    "evaluator": {
        "func": "check_include_exclude",
        "result": {
            "type": "vm_command_line",
            "command": "which spotify"
        },
        "expected": {
            "type": "rule",
            "rules": {
                "include": ["spotify"],
                "exclude": ["not found"]
            }
        }
    }
}

try:
    env = DesktopEnv(action_space="pyautogui", provider_name = 'docker', os_type = 'Windows', require_a11y_tree = True)
except Exception as e:
    print(f"Error: {e}.")

try: 
    env = DesktopEnv(
            provider_name="vmware",
            path_to_vm="C:\\Users\\dylan\\Synth-Dataset\\vmware_vm_data\\Ubuntu0\\Ubuntu0.vmx",
            snapshot_name="init_state",  # Make sure this snapshot exists
            action_space="pyautogui",
            require_a11y_tree=True
        )

    # obs = env.reset(task_config=example)
    obs, reward, done, info = env.step("pyautogui.rightClick()")
except Exception as e:
    print(f"Error: {e}.")
    raise

import base64
def encode_image_from_variable(image_content):
    return base64.b64encode(image_content).decode('utf-8')

base64_image = encode_image_from_variable(obs['screenshot'])

#%%
SYS_TASK_INIT = """What does this screen show? Imagine you are a real user on this webpage. Given the webpage link, please provide a single task that a user might perform on this website and the corresponding first action towards completing that task. You can use any software in the computer and the web. Be creative and come up with diverse tasks. The task should be simple enough that can be finished in a few steps. You should propose tasks that are clear and specific.

Task proposal rules:
1. The website should be explicitly mentioned in the task description.
2. The task should be specific and clear.
3. The task should be achievable within a few steps.
4. The task should be relevant to the content of the webpage.
5. You should only propose tasks that do not require login to execute the task
6. Provide concrete information or constraints to the task, and use mock-up information
(identifier, number, personal information, name, attributes, etc.) to make the task more specific and realistic.
7. The task description should provide all the necessary information to complete the task.

Output with JSON block:
```{"task":"<TASK>", "action":"<ACTION>"}```
"""

SYS_TASK_INIT_LOCAL = """What does this screen show? Imagine you are a real user on this webpage. Given the software, please provide a single task that a user might perform using this software and the corresponding first action towards completing that task. You can use any tool in the computer and the web. Be creative and come up with diverse tasks. The task should be simple enough that can be finished in a few steps. You should propose tasks that are clear and specific.

Task proposal rules:
1. The software should be explicitly mentioned in the task description.
2. The task should be specific and clear.
3. The task should be achievable within a few steps.
4. The task should be relevant to the content of the webpage.
5. You should only propose tasks that do not require login to execute the task
6. Provide concrete information or constraints to the task, and use mock-up information
(identifier, number, personal information, name, attributes, etc.) to make the task more specific and realistic.
7. The task description should provide all the necessary information to complete the task.

Output with JSON block:
```{"task":"<TASK>", "action":"<ACTION>"}```
"""

SYS_TASK_FOLLOWUP = """What does this screen show? Imagine you are a real user on this webpage. Given the webpage link, and the tasks the user has done, please provide a single followup task that a user might perform on this website and the corresponding first action towards completing that task. You can use any software in the computer and the web. Be creative and come up with diverse tasks. The task should be simple enough that can be finished in a few steps.

Task proposal rules:
1. The website should be explicitly mentioned in the task description.
2. The task should depends on the previous tasks.
2. The task should be specific and clear.
3. The task should be achievable within a few steps.
4. The task should be relevant to the content of the webpage.
5. You should only propose tasks that do not require login to execute the task
6. Provide concrete information or constraints to the task, and use mock-up information
(identifier, number, personal information, name, attributes, etc.) to make the task more specific and realistic.
7. The task description should provide all the necessary information to complete the task.

Output with JSON block:
```{"task":"<TASK>", "action":"<ACTION>"}```
"""

SYS_TASK_FOLLOWUP_LOCAL = """What does this screen show? Imagine you are a real user on this computer. Given the software, and the tasks the user has done, please provide a single followup task that a user might perform on this computer and the corresponding first action towards completing that task. You can use any tool in the computer and the web. Be creative and come up with diverse tasks. The task should be simple enough that can be finished in a few steps.

Task proposal rules:
1. The task should depend on the previous tasks.
2. The task should be specific and clear.
3. The task should be achievable within a few steps.
4. The task should be relevant to the content of the software.
5. You should only propose tasks that do not require login to execute the task
6. Provide concrete information or constraints to the task, and use mock-up information
(identifier, number, personal information, name, attributes, etc.) to make the task more specific and realistic.
7. The task description should provide all the necessary information to complete the task.

Output with JSON block:
```{"task":"<TASK>", "action":"<ACTION>"}```
"""

SYS_TASK_ACTION = """You are an agent which follow my instruction and perform desktop computer tasks as instructed.You have good knowledge of computer and good internet connection and assume your code will run on a computer for controlling the mouse and keyboard. For each step, you will be asked to finish a task, and you will get an observation of an image, which is the screenshot of the computer screen. You also know the actions that are already done towards the target. If you are on the lock screen, note that the password for the user is \"password\". You need to predict the next action of the computer based on the image. If you have tried similar actions several times but haven't success, analyze the reason carefully and propose a different action. Try to think if you were wrong or you missed any steps. If you think the task is finished, return "DONE" in the action field.

Rules:
1. First analyze the screenshot carefully, pay attention to details in the screenshot like popups, etc. 
2. Then analyze the previous thoughts and actions to make sure you are on the right track.
3. If you find you have tried similar actions several times but haven't success, analyze the reason carefully. Try to think if you were wrong or you missed any steps. Carefully analyze the screenshot to find the reason of the failure and propose a different action.
4. If you think the task is finished, return "DONE" in the action field.

Output with JSON block:
```{"thoughts":"<THOUGHTS and REASONS>", "action":"<ACTION>"}```
"""

SYS_COMPUTER_ACTION = """You are an agent which follow my instruction and perform desktop computer tasks as instructed.You have good knowledge of computer and good internet connection and assume your code will run on a computer for controlling the mouse and keyboard.For each step, you will be asked to finish a task, and you will get an observation of an image, which is the screenshot of the computer screen. You need to predict the next action of the computer based on the image. You have all the permissions to proceed and you don't need to ask for permission. The safety checks are acknowledged and you can always proceed."""

#%%
website = "sustainablecooks.com"
website = "berkeley.edu/"
software = 'VS Code'

def call_gpt_4o(sys_prompt, user_prompt, img):
    payload = {
        "model": "gpt-4o", 
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
        headers=headers,
        json=payload
    )
    # breakpoint()
    return response.json()['output'][0]['content'][0]['text']



def call_o3(sys_prompt, user_prompt, img):
    payload = {
        "model": "o3-mini", 
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
        headers=headers,
        json=payload
    )

    return response.json()['output']


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
        headers=headers,
        json=payload
    )
    # breakpoint()

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
            print("Failed to parse JSON:", e)
            return None
    elif match_2:
        json_str = match_2.group(0)
        try:
            data = json.loads(json_str)
            print("Extracted JSON:", data)
            return data
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            return None
    else:
        print("No JSON block found.", llm_output)
        return None
    
def initial_task_propose(website):
    sys_prompt = SYS_TASK_INIT
    user_prompt = f"Given the website {website}, what task would a user perform?"
    while True:
        try:
            llm_output = call_gpt_4o(sys_prompt, user_prompt, base64_image)
            task_info = parse_json(llm_output)['task']
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            continue

    return task_info


def initial_task_propose_local(software):
    sys_prompt = SYS_TASK_INIT_LOCAL
    user_prompt = f"Given the software {software}, what task would a user perform?"
    while True:
        try:
            llm_output = call_gpt_4o(sys_prompt, user_prompt, base64_image)
            task_info = parse_json(llm_output)['task']
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            continue

    return task_info

def followup_task_propose(website, task_history, img):
    sys_prompt = SYS_TASK_FOLLOWUP
    user_prompt = f"Given the website {website} and the task history {task_history}, what would be a followup task?"
    while True:
        try:
            llm_output = call_gpt_4o(sys_prompt, user_prompt, img)
            task_info = parse_json(llm_output)['task']
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            continue
    return task_info

def followup_task_propose_local(software, task_history, img):
    sys_prompt = SYS_TASK_FOLLOWUP_LOCAL
    user_prompt = f"Given the software {software} and the task history {task_history}, what would be a followup task?"
    while True:
        try:
            llm_output = call_gpt_4o(sys_prompt, user_prompt, img)
            task_info = parse_json(llm_output)['task']
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            continue
    return task_info


def generate_action(task, thoughts_history, action_history, a11y_tree, img,
                    model='gpt-4o'):
    # print(a11y_tree)
    sys_prompt = SYS_TASK_ACTION
    
    user_prompt = f"""Given the task: {task}. Here is your previous thinking process to complete the task {thoughts_history}. 
    Here is your previous actions tried {action_history}. 
    Attached is the current screenshot, what would be the next action?"""
    match model:
        case 'gpt-4o':
            llm_method = call_gpt_4o
        case 'o3-mini':
            llm_method = call_o3
    
    while True:
        try:
            llm_output = llm_method(sys_prompt, user_prompt, img)
            # breakpoint()
            parsed_output = parse_json(llm_output)
            action_info = parsed_output['action']
            thoughts = parsed_output['thoughts']
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            continue

    return action_info, thoughts

def generate_computer_use_action(task, step, command_history, img, iter=0, max_iter=3):
    sys_prompt = SYS_COMPUTER_ACTION
    user_prompt = f"Given the task: {task}, I have done the following actions: {command_history}. Next, I need to do the step: {step}. What would be the action?"
    if iter < max_iter:
        try:
            llm_output = call_computer_use_preview(sys_prompt, user_prompt, img)
            action_info = parse_computer_use_preview(llm_output)
            if action_info is None:
                raise Exception("Action was not parsed from the response")
            return action_info
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            return generate_computer_use_action(task, step, command_history, img, iter+1)
    else:
        raise Exception(f"Failed to generate computer use action after {max_iter} attempts")

def encode_image_from_variable(image_content):
    return base64.b64encode(image_content).decode('utf-8')


# %%
# task = initial_task_propose(website)
possible_local_software = [
    'VS Code', 'Chrome', 'ThunderBird Mail', 'LibreOffice Writer',
    'LibreOffice Calc', 'LibreOffice Impress', 'Files',
    'Calendar', 'AisleRiot Solitare'
]
software = random.choice(possible_local_software)
task = initial_task_propose_local(software)
thoughts_history = []
action_history = []
command_history = []
a11y_tree = obs['accessibility_tree']
#%%
def solve_task(task, env, base64_image, a11y_tree, time_btw_tasks=5, max_steps=15):
    for i in tqdm(range(max_steps)):
        action, thoughts = generate_action(task, thoughts_history, action_history, a11y_tree, base64_image)
        thoughts_history.append(thoughts)
        action_history.append(action)
        if action == 'DONE':
            return True # successful completion
        python_command = generate_computer_use_action(task, action, command_history, base64_image)
        command_history.append(python_command)
        python_command += f'; time.sleep({time_btw_tasks})'
        print(python_command)
        obs, reward, done, info = env.step(python_command)
        # print(info)
        base64_image = encode_image_from_variable(obs['screenshot'])
        a11y_tree = obs['accessibility_tree']

        image = Image.open(BytesIO(obs['screenshot']))
        plt.imshow(image)
    return False

completed = solve_task(task, env, base64_image, a11y_tree, time_btw_tasks=1)

# %%
# task = followup_task_propose(website, task_history, base64_image)
num_iterations = 5
trajectories = []
for i in range(num_iterations):
    if completed:
        trajectories.append((software, task, action_history, thoughts_history))
        task_history.append(task)
        task = followup_task_propose_local(software, task_history, base64_image)
        thoughts_history = []
        action_history = []
        command_history = []
        # obs, reward, done, info = env.reset(task_config=task)
        # time.sleep(20) # wait for reset
        null_action = 'pyautogui.moveTo(960, 540); time.sleep(0.5)'
        # move the mouse to the center of the screen
        obs, reward, done, info = env.step(null_action)
        base64_image = encode_image_from_variable(obs['screenshot'])
        a11y_tree = obs['accessibility_tree']
        completed = solve_task(task, env, base64_image, a11y_tree, time_btw_tasks=1)
    else:
        completed = solve_task(task, env, base64_image, a11y_tree, time_btw_tasks=1)
    print(f"Run {i+1} of {num_iterations} completed: {completed}")

# Write trajectories to file
with open('trajectories.jsonl', 'w') as f:
    for trajectory in trajectories:
        f.write(json.dumps({"software": trajectory[0], "task": trajectory[1], "action_history": trajectory[2], "thoughts_history": trajectory[3]}) + "\n")

# %%
# for i in range(10):
#     action, thoughts = generate_action(task, thoughts_history, base64_image)
#     thoughts_history.append(thoughts)
#     # thoughts_history = thoughts_history[-10:]
#     if action == 'DONE':
#         break
#     python_command = generate_computer_use_action(task, action, base64_image)
#     python_command += '; time.sleep(5)'
#     print(python_command)
#     obs, reward, done, info = env.step(python_command)
#     base64_image = encode_image_from_variable(obs['screenshot'])

#     image = Image.open(BytesIO(obs['screenshot']))
#     plt.figure()
#     plt.imshow(image, cmap='gray')
# %%
