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
