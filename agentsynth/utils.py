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

# Local LLaVa imports
try:
    import torch
    from transformers import LlavaForConditionalGeneration, LlavaProcessor
    LOCAL_LLAVA_AVAILABLE = True
except ImportError:
    LOCAL_LLAVA_AVAILABLE = False
    print("Warning: Local LLaVa dependencies not available. Install transformers and torch to use local inference.")

#%%

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
USE_LOCAL_LLAVA = os.getenv('USE_LOCAL_LLAVA', 'false').lower() == 'true'
LOCAL_LLAVA_MODEL_PATH = os.getenv('LOCAL_LLAVA_MODEL_PATH', 'llava-hf/llava-1.5-7b-hf')
LOCAL_LLAVA_CHECKPOINT_PATH = os.getenv('LOCAL_LLAVA_CHECKPOINT_PATH', None)  # Path to LoRA checkpoint
LOCAL_LLAVA_DEVICE = os.getenv('LOCAL_LLAVA_DEVICE', 'auto')  # 'auto', 'cuda', 'cpu'

# Global variables for local model
_local_llava_model = None
_local_llava_processor = None

headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }

def load_local_llava_model():
    """Load the local LLaVa model and processor"""
    global _local_llava_model, _local_llava_processor
    
    if not LOCAL_LLAVA_AVAILABLE:
        raise ImportError("Local LLaVa dependencies not available. Install transformers and torch.")
    
    if _local_llava_model is None or _local_llava_processor is None:
        print(f"Loading local LLaVa model: {LOCAL_LLAVA_MODEL_PATH}")
        
        # Determine device
        if LOCAL_LLAVA_DEVICE == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = LOCAL_LLAVA_DEVICE
        
        print(f"Using device: {device}")
        
        # Load processor
        _local_llava_processor = LlavaProcessor.from_pretrained(LOCAL_LLAVA_MODEL_PATH)
        
        # Load model
        _local_llava_model = LlavaForConditionalGeneration.from_pretrained(
            LOCAL_LLAVA_MODEL_PATH,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map=device if device == 'cuda' else None,
            low_cpu_mem_usage=True
        )
        
        # Load LoRA checkpoint if specified
        if LOCAL_LLAVA_CHECKPOINT_PATH and os.path.exists(LOCAL_LLAVA_CHECKPOINT_PATH):
            print(f"Loading LoRA checkpoint from: {LOCAL_LLAVA_CHECKPOINT_PATH}")
            try:
                from peft import PeftModel
                _local_llava_model = PeftModel.from_pretrained(_local_llava_model, LOCAL_LLAVA_CHECKPOINT_PATH)
                print("LoRA checkpoint loaded successfully")
            except ImportError:
                print("Warning: PEFT not available. Cannot load LoRA checkpoint.")
            except Exception as e:
                print(f"Warning: Failed to load LoRA checkpoint: {e}")
        
        if device == 'cpu':
            _local_llava_model = _local_llava_model.to(device)
        
        print("Local LLaVa model loaded successfully")
    
    return _local_llava_model, _local_llava_processor

def call_local_llava(sys_prompt, user_prompt, img, model_name="local-llava"):
    """Call local LLaVa model for inference"""
    if not LOCAL_LLAVA_AVAILABLE:
        raise ImportError("Local LLaVa dependencies not available")
    
    model, processor = load_local_llava_model()
    
    # Prepare images
    if type(img) == list:
        images = []
        for item in img:
            image_data = base64.b64decode(item)
            image = Image.open(BytesIO(image_data))
            images.append(image)
    else:
        image_data = base64.b64decode(img)
        image = Image.open(BytesIO(image_data))
        images = [image]
    
    # Enhanced system prompt with stronger JSON formatting instructions
    enhanced_sys_prompt = sys_prompt + """

CRITICAL: You MUST respond with ONLY a valid JSON block. Do not include any text before or after the JSON. 
The JSON must be properly formatted and complete. Start your response with { and end with }.

Example format:
{"thoughts": "Your detailed thoughts here", "action": "Your action here"}

Do not write any explanation or text outside the JSON block."""
    
    # Prepare conversation
    conversation = [
        {
            "role": "system",
            "content": enhanced_sys_prompt
        },
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": images[0] if len(images) == 1 else images}
            ]
        }
    ]
    
    # Apply chat template
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    
    # Fix: Manually prepend system prompt if it's missing
    if not prompt.startswith(enhanced_sys_prompt[:50]):  # Check if system prompt is included
        prompt = f"{enhanced_sys_prompt}\n\n{prompt}"
    
    # System prompt is now properly included in the prompt
    
    # Process inputs
    inputs = processor(text=prompt, images=images, return_tensors="pt")
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate with more conservative settings for better JSON output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,  # Lower temperature for more deterministic output
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            repetition_penalty=1.1  # Reduce repetition
        )
    
    # Decode only the new tokens (generated response)
    input_length = inputs['input_ids'].shape[1]
    new_tokens = outputs[0][input_length:]
    response = processor.decode(new_tokens, skip_special_tokens=True)
    
    # Clean up response - remove any leading/trailing whitespace
    response = response.strip()
    
    # Try to extract JSON from response if it's not already in JSON format
    if not response.startswith('{'):
        # Look for JSON block in the response
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            response = json_match.group(0)
        else:
            # If no JSON found, try to create a basic JSON response
            response = '{"thoughts": "' + response.replace('"', '\\"') + '", "action": "DONE"}'
    
    # Clean up JSON by fixing common issues
    import re
    import json
    
    # Remove control characters that are not allowed in JSON
    response = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', response)
    
    # Try to parse and fix the JSON
    try:
        # Test if the JSON is valid
        parsed = json.loads(response)
        # If valid, re-encode to ensure proper formatting
        response = json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError as e:
        # If JSON is invalid, try to fix common issues
        try:
            # Fix unescaped newlines in string values
            fixed_response = re.sub(r'(?<!\\)\n', '\\n', response)
            # Fix unescaped quotes in string values (but be careful not to break the JSON structure)
            fixed_response = re.sub(r'(?<!\\)"(?![^}]*})', '\\"', fixed_response)
            
            # Try parsing again
            parsed = json.loads(fixed_response)
            response = json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            # If still invalid, create a safe fallback
            safe_response = response.replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
            response = f'{{"thoughts": "JSON parsing error, original response: {safe_response}", "action": "DONE"}}'
    
    # Response is now properly extracted (only new tokens)
    
    # Log token usage (approximate)
    input_tokens = inputs['input_ids'].shape[1]
    output_tokens = outputs.shape[1] - input_tokens
    total_tokens = outputs.shape[1]
    
    current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('token_count_llm.txt', 'a') as f:
        f.write(f"local-llava: {current_date_time} {model_name} {total_tokens}\n")
    
    return response

def call_llms(sys_prompt, user_prompt, img, model = 'gpt-4.1'):
    """Unified LLM calling function that supports both OpenAI and local LLaVa"""
    if USE_LOCAL_LLAVA and LOCAL_LLAVA_AVAILABLE:
        return call_local_llava(sys_prompt, user_prompt, img, model)
    
    # Original OpenAI implementation
    if type(img) == list:
        screenshots_payload = [{"type": "image_url", "image_url": {'url':f"data:image/png;base64,{item}"}} for item in img]
    else:
        screenshots_payload = [{"type": "image_url", "image_url": {'url':f"data:image/png;base64,{img}"}}]

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

    current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('token_count_llm.txt', 'a') as f:
        f.write(f"gpt: {current_date_time} {model} {response.usage.total_tokens}\n")
    return response.choices[0].message.content



def call_gpt(sys_prompt, user_prompt, img, model = 'gpt-4.1'):
    """GPT calling function that supports both OpenAI and local LLaVa"""
    if USE_LOCAL_LLAVA and LOCAL_LLAVA_AVAILABLE:
        return call_local_llava(sys_prompt, user_prompt, img, model)
    
    # Original OpenAI implementation
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
    with open('token_count_gpt.txt', 'a') as f:
        f.write(f"gpt: {current_date_time} {model} {response.json()['usage']['total_tokens']}\n")
    return response.json()['output'][0]['content'][0]['text']






def call_computer_use_preview(sys_prompt, user_prompt, img, model = "computer-use-preview"):
    """Computer use preview function - only supports OpenAI API, not local LLaVa"""
    if USE_LOCAL_LLAVA and LOCAL_LLAVA_AVAILABLE:
        print("Warning: Computer use preview is not available with local LLaVa. Falling back to regular LLM call.")
        # Fall back to regular LLM call for local inference
        response_text = call_local_llava(sys_prompt, user_prompt, img, model)
        # Create a mock response structure for compatibility
        return {
            'output': [{
                'type': 'text',
                'content': response_text
            }],
            'usage': {'total_tokens': len(response_text.split())}  # Approximate
        }
    
    # Original OpenAI implementation
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
    with open('token_count_computer.txt', 'a') as f:
        f.write(f"computer: {current_date_time} {model} {response.json()['usage']['total_tokens']}\n")
    # print('computer use total tokens: ', response.json()['usage']['total_tokens'])
    return response.json()


def parse_computer_use_preview(response):
    raw_output = response['output']
    for item in raw_output:
        if item['type'] == 'computer_call':
            action = item['action']
            return parse_computer_use_pyautogui(action)
        # TODO: create a better way to derive computer use action from text response from local LLaVa
        # elif item['type'] == 'text' and USE_LOCAL_LLAVA:
        #     # Handle fallback case for local LLaVa
        #     print("Using text response from local LLaVa for computer use action.")
        #     # Try to extract action from text response
        #     text_content = item['content']
        #     # Simple fallback - return a basic action
        #     return f"# Local LLaVa response: {text_content}"
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


def followup_task_propose_persona(persona, task_history, img, failed_task = None):
    sys_prompt = SYS_TASK_FOLLOWUP_PERSONA
    user_prompt = f"You are {persona}. Given the task history {task_history}, what would be a followup task?"
    if failed_task:
        user_prompt += f" Note that these tasks {failed_task} are too hard for the agent, propose a simplier one."

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
            llm_output = call_gpt(sys_prompt, user_prompt, img)
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
    user_prompt = f"Given the task {task}, the key points to finish the task {key_points}, and the screenshot history, is the agent successful?"
    while True:
        try:
            llm_output = call_gpt(sys_prompt, user_prompt, img, model)
            verifier_info = parse_json(llm_output)
            thoughts = verifier_info['thoughts']
            success = verifier_info['success']
            success_rate = verifier_info['success rate']
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(10)
            continue

    return success_rate, success, thoughts


def generate_verifier_verdict_key_info(task, img, model = 'gpt-4.1'):
    key_points, _ = generate_verifier_key_points(task, model)
    necessary_list = []
    for item in img:
        necessary, _ = generate_verifier_key_screen(task, key_points, item, model)
        necessary_list.append(necessary)
    
    img_list_for_verdict = [item for item, necessary in zip(img, necessary_list) if necessary]
    success_rate, success, thoughts = generate_verifier_verdict(task, key_points, img_list_for_verdict, model)
    return success_rate, success, thoughts, necessary_list



def encode_image_from_variable(image_content):
    if image_content is None:
        raise ValueError("Image content is None - cannot encode. This usually means the environment didn't provide a screenshot.")
    if not isinstance(image_content, (bytes, bytearray)):
        raise ValueError(f"Image content must be bytes, got {type(image_content)}")
    return base64.b64encode(image_content).decode('utf-8')

def decode_image_from_variable(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


def select_persona():
    with open('persona.jsonl', 'r', encoding='utf-8') as file:
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
