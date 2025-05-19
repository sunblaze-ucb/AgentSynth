#%%
import json
import requests

#%%
def parse_computer_use_pyautogui(data):
    action_type = data.get('type')
    if action_type == 'click':
        button = data.get('button')
        x = data.get('x')
        y = data.get('y')
        return "pyautogui.click(x={}, y={}, button='{}')".format(x, y, button)
    
    elif action_type == 'double_click':
        x = data.get('x')
        y = data.get('y')
        return "pyautogui.doubleClick(x={}, y={})".format(x, y)
    
    elif action_type == 'move':
        x = data.get('x')
        y = data.get('y')
        return "pyautogui.moveTo(x={}, y={})".format(x, y)
    
    elif action_type == 'type':
        text = data.get('text')
        return """pyautogui.write({})""".format(repr(text))
    
    elif action_type == 'scroll':
        amount = -data.get('scroll_y')
        x = data.get('x')
        y = data.get('y')
        return "pyautogui.click(x={}, y={}); pyautogui.scroll({})".format(x, y, amount)
    
    elif action_type == 'drag':
        path = data.get('path')
        x1 = path[0]['x']
        y1 = path[0]['y']
        x2 = path[-1]['x']
        y2 = path[-1]['y']
        return "pyautogui.moveTo(x={}, y={}); time.sleep(1); pyautogui.dragTo(x={}, y={})".format(x1, y1, x2, y2)


    elif action_type == 'keypress':
        keys = data.get('keys')
        keys = [key.lower() for key in keys]  # Convert to lowercase
        if len(keys) > 1:
            return f"pyautogui.hotkey({', '.join(repr(k) for k in keys)})"
        return "pyautogui.press({})".format(keys)
    
    elif action_type == 'wait':
        return "time.sleep(1)"
    
    else:
        print(f"Unknown action type: {action_type}")
        return "time.sleep(1)"
    

#%%
if __name__ == "__main__":
    OPENAI_API_KEY = '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'

    import base64
    base64_image = image_path = "E:\\Desktop2025.1.17\\CS 294 LLM Agent\\osworld\\OSWorld\\results\\" \
        "pyautogui\\screenshot\\gpt-4o\\chrome\\bb5e4c0d-f964-439c-97b6-bdb9747de3f4\\step_1_20250316@184639.png"

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = encode_image(image_path)

    headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENAI_API_KEY}"
                }

    payload = {
        "model": "computer-use-preview", 
        "top_p": 0.9, 
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
                "role": "user", 
                "content": [
                    {
                        "type": "input_text", 
                        "text": "Generate a Press Enter key action"
                    },
                    {
                        "type": "input_image", 
                        "image_url": f"data:image/png;base64,{base64_image}"
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

    response.json()
    raw_output = response.json()['output']
    for item in raw_output:
        if item['type'] == 'computer_call':
            action = item['action']
    print(action)
    parse_computer_use_pyautogui(action)
# %%
