#%%
import sys
import os
from desktop_env.desktop_env import DesktopEnv
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import json
from datetime import datetime
import time

from utils import initial_task_propose_persona, followup_task_propose_persona,  generate_action, generate_computer_use_action, generate_summary, generate_verifier, select_persona, encode_image_from_variable, decode_image_from_variable, generate_key_info, generate_subtask_summary
from prompts import EXAMPLE

#%%
def execute_task(task, env, base64_image, info_history = [], max_steps=10):
    thoughts_history = []
    action_history = []
    command_history = []
    base64_img_list = []
    img_list = []

    for i in range(max_steps):
        action, thoughts = generate_action(task, thoughts_history, action_history, info_history, base64_image)
        thoughts_history.append(thoughts)
        action_history.append(action)

        if action == 'DONE':
            if len(base64_img_list) == 0:
                obs, reward, done, info = env.step("time.sleep(1)")
                base64_image = encode_image_from_variable(obs['screenshot'])
                base64_img_list.append(base64_image)
                img = Image.open(BytesIO(obs['screenshot']))
                img_list.append(img)
            return True, thoughts_history, action_history, command_history, base64_img_list, img_list

        python_command = generate_computer_use_action(task, action, command_history, base64_image)
        command_history.append(python_command)
        if python_command != None:
            python_command += '; time.sleep(5)'
        else:
            python_command = 'time.sleep(5)'
        print(f'action: {i}', python_command)

        obs, reward, done, info = env.step(python_command)
        base64_image = encode_image_from_variable(obs['screenshot'])
        base64_img_list.append(base64_image)

        img = Image.open(BytesIO(obs['screenshot']))
        img_list.append(img)

    return False, thoughts_history, action_history, command_history, base64_img_list, img_list


def update_trace(ret, task_history, task, thoughts, actions, commands, b64_list, info_history, failure_task_history, data_save):
    if ret:
        task_history.append(task)
        key_info = generate_key_info(task, thoughts, b64_list)
        info_history.append(key_info)

        data_save['thoughts'].append(thoughts)
        data_save['actions'].append(actions)
        data_save['commands'].append(commands)
        data_save['screenshots'].append(b64_list)
        data_save['done'].append(True)
    else:
        task = generate_subtask_summary(b64_list)
        if task is not None:
            task_history.append(task)
            key_info = generate_key_info(task, thoughts, b64_list)
            info_history.append(key_info)

            data_save['thoughts'].append(thoughts)
            data_save['actions'].append(actions)
            data_save['commands'].append(commands)
            data_save['screenshots'].append(b64_list)
            data_save['done'].append(False)
        else:
            failure_task_history.append(task)
    
    return task_history, info_history, data_save, failure_task_history

#%%
if __name__ == "__main__":
    for i in range(5):
        try:
            while True:
                with open('lock.txt', 'r') as lock_file:
                    lock_status = lock_file.read().strip()
                if lock_status == 'False':
                    with open('lock.txt', 'w') as lock_file:
                        lock_file.write('True')
                    break
                time.sleep(5)

            env = DesktopEnv(action_space="pyautogui", provider_name = 'docker', os_type = 'Ubuntu', require_a11y_tree = False)

            obs = env.reset(task_config = EXAMPLE)
            while True:
                try:
                    obs, reward, done, info = env.step("")
                    base64_image = encode_image_from_variable(obs['screenshot'])

                    image = Image.open(BytesIO(obs['screenshot']))
                    plt.imshow(image)
                    with open('lock.txt', 'w') as lock_file:
                        lock_file.write('False')
                    break
                except Exception as e:
                    print(e)
                    time.sleep(5)

            task_history = []
            task_history_original = []
            failure_task_history = []
            info_history = []

            persona = select_persona()
            print(persona)
            task = initial_task_propose_persona(persona, base64_image)
            task_history_original.append(task)

            data_save = {'thoughts': [], 'actions': [], 'commands': [], 'screenshots': [], 'done': []}

            ret, thoughts, actions, commands, b64_list, img_list =  execute_task(task, env, base64_image, info_history)

            task_history, info_history, data_save, failure_task_history = update_trace(ret, task_history, task, thoughts, actions, commands, b64_list, info_history, failure_task_history, data_save)

            for i in range(5):
                print('Processing task:', i)
                task = followup_task_propose_persona(persona, task_history, b64_list[-1], failure_task_history)
                task_history_original.append(task)

                ret, thoughts, actions, commands, b64_list, img_list =  execute_task(task, env, b64_list[-1], info_history)

                task_history, info_history, data_save, failure_task_history = update_trace(ret, task_history, task, thoughts, actions, commands, b64_list, info_history, failure_task_history, data_save)
        
            summary_task = generate_summary(task_history, b64_list[-1])

            data_save['task_history'] = task_history
            data_save['task_history_original'] = task_history_original
            data_save['failure_task_history'] = failure_task_history
            data_save['info_history'] = info_history
            data_save['persona'] = persona
            data_save['summary_task'] = summary_task


            task_levels = [task_history[0]]
            for i in range(2, len(task_history) + 1):
                temp = generate_summary(task_history[:i], data_save['screenshots'][i - 1][-1])
                task_levels.append(temp)

            data_save['task_levels'] = task_levels

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"data_save_{timestamp}.json"
            folder = ''
            with open(folder + filename, 'w') as f:
                json.dump(data_save, f, indent=4)
            
            env.close()

        except Exception as e:
            print(e)
            time.sleep(120)
            try:
                env.close()
            except:
                pass

# %%
