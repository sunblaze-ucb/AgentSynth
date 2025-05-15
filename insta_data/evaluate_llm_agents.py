import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

# Import the generate_trajectory function from pipeline
from insta.pipeline import (
    generate_trajectory, 
    DEFAULT_MAX_ACTIONS,
    DEFAULT_AGENT_RESPONSE_KEY
)

# Import necessary configs and classes
from insta.configs import (
    get_browser_config,
    get_agent_config,
    get_judge_config
)

from insta.gym_env import InstaEnv

with open("secrets.json", "r") as f:
    secrets = json.load(f)

def get_client_kwargs_for_model(model_name):
    """
    Returns the appropriate client_kwargs based on the model name.
    
    Args:
        model_name (str): The name of the model
        
    Returns:
        dict: Client configuration kwargs
    """
    # For Claude models
    if model_name.startswith("claude"):
        return {
            "api_key": secrets.get("ANTHROPIC_API_KEY"),
            "base_url": "https://api.anthropic.com/v1"
        }
    # For Gemini models
    elif model_name.startswith("gemini"):
        return {
            "api_key": secrets.get("GOOGLE_API_KEY"),
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai"
        }
    # For OpenAI models (default)
    else:
        return {
            "api_key": secrets.get("OPENAI_API_KEY")
            # No base_url needed for OpenAI (it uses the default)
        }

def evaluate_task_with_trajectory(task_data, model_name):
    """
    Evaluates a single task using generate_trajectory from pipeline.py
    Returns a boolean indicating success.
    """
    # Extract task text and website
    task_text = task_data.get("task") or task_data.get("description", "")
    website = task_data.get("website", "")
    print(f"Evaluating task for website '{website}' using {model_name}: {task_text}")
    
    try:
        # Configure the agent based on the model name
        client_kwargs = get_client_kwargs_for_model(model_name)
        
        agent_config = get_agent_config(
            client_kwargs=client_kwargs,
            generation_kwargs={"model": model_name}
        )
        judge_config = get_judge_config(
            client_kwargs=client_kwargs,
            generation_kwargs={"model": model_name}
        )
        browser_config = get_browser_config()
        
        # Create environment
        env = InstaEnv(config=browser_config)
        
        # Generate trajectory
        observations, actions, judgment = generate_trajectory(
            agent=agent_config,
            judge=judge_config,
            env=env,
            url=website,
            instruction=task_text,
            max_actions=10,
            agent_response_key=DEFAULT_AGENT_RESPONSE_KEY
        )
        
        # Check if task was successful based on judgment
        success = False
        if judgment and isinstance(judgment, dict):
            success_value = judgment.get('success', 0.0)
            task_success = judgment.get('task_success', False)
            success = (success_value >= 0.9) or task_success
        
        print(f"Result for model {model_name}: {'Success' if success else 'Failure'}")
        
        # Clean up
        if hasattr(env, 'close'):
            env.close()
            
        return success
        
    except Exception as e:
        print(f"Error evaluating task with {model_name}: {e}")
        return False


def evaluate_sequence_with_trajectory(tasks, model_name):
    """
    Evaluates a sequence of tasks using generate_trajectory.
    Only evaluates a task if all previous tasks succeeded.
    """
    sequence_result = []
    
    # Configure once for the whole sequence
    client_kwargs = get_client_kwargs_for_model(model_name)
    
    agent_config = get_agent_config(
        client_kwargs=client_kwargs,
        generation_kwargs={"model": model_name}
    )
    judge_config = get_judge_config(
        client_kwargs=client_kwargs,
        generation_kwargs={"model": model_name}
    )
    browser_config = get_browser_config()
    
    # Create environment once
    env = InstaEnv(config=browser_config)
    
    try:
        for idx, task in enumerate(tasks):
            if idx > 0 and not sequence_result[-1]:
                # Skip remaining tasks once a task has failed
                print(f"Skipping task {idx+1} in sequence because a previous task failed.")
                sequence_result.append(False)
                continue
            
            task_text = task.get("task") or task.get("description", "")
            website = task.get("website", "")
            
            try:
                # Generate trajectory
                observations, actions, judgment = generate_trajectory(
                    agent=agent_config,
                    judge=judge_config,
                    env=env,
                    url=website,
                    instruction=task_text,
                    max_actions=10,
                    agent_response_key=DEFAULT_AGENT_RESPONSE_KEY
                )
                
                # Check if task was successful
                success = False
                if judgment and isinstance(judgment, dict):
                    success_value = judgment.get('success', 0.0)
                    task_success = judgment.get('task_success', False)
                    success = (success_value >= 0.9) or task_success
                
                sequence_result.append(success)
                
            except Exception as e:
                print(f"Error in task {idx+1}: {e}")
                sequence_result.append(False)
    
    finally:
        # Clean up
        if hasattr(env, 'close'):
            env.close()
    
    return sequence_result


def load_jsonl(filename):
    """Loads a JSONL file and returns a list of JSON objects."""
    tasks = []
    with open(filename, "r", encoding="utf8") as infile:
        for line in infile:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def group_tasks_by_sequence(tasks):
    """
    Groups tasks into sequences where consecutive tasks share the same website.
    """
    if not tasks:
        return []
    sequences = []
    current_sequence = [tasks[0]]
    current_website = tasks[0].get("website")
    
    for task in tasks[1:]:
        website = task.get("website")
        if website == current_website:
            current_sequence.append(task)
        else:
            sequences.append(current_sequence)
            current_sequence = [task]
            current_website = website
    sequences.append(current_sequence)
    return sequences


def save_result_to_jsonl(result_entry, filename):
    """Appends a single result entry to a JSONL file."""
    with open(filename, "a", encoding="utf8") as outfile:
        json.dump(result_entry, outfile)
        outfile.write("\n")


def load_results_from_jsonl(filename):
    """
    Loads results from a JSONL file and organizes them into the expected structure.
    """
    try:
        results = {}
        with open(filename, "r", encoding="utf8") as infile:
            for line in infile:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    model = entry.get("model")
                    task_idx = entry.get("task_idx")
                    success = entry.get("success")
                    
                    if model not in results:
                        results[model] = []
                    
                    # Extend the results list if needed
                    while len(results[model]) <= task_idx:
                        results[model].append(None)
                    
                    results[model][task_idx] = success
        return results
    except FileNotFoundError:
        return None


def save_final_results(results, filename):
    """Saves the final consolidated results to a JSON file."""
    with open(filename, "w", encoding="utf8") as outfile:
        json.dump(results, outfile, indent=2)
    print(f"Final results saved to {filename}")


def main():
    # List of models to evaluate
    models_to_evaluate = ["gpt-4.1", "o4-mini"]
    
    # Result filenames
    summarized_results_jsonl = "summarized_results_trajectory.jsonl"
    sequence_results_jsonl = "sequence_results_trajectory.jsonl"
    baseline_results_jsonl = "baseline_results_trajectory.jsonl"
    
    # Final consolidated result filenames
    summarized_results_file = "summarized_results_trajectory_final.json"
    sequence_results_file = "sequence_results_trajectory_final.json"
    baseline_results_file = "baseline_results_trajectory_final.json"

    # Create output directories if they don't exist
    os.makedirs("results", exist_ok=True)
    
    # ========== Evaluate Summarized Task Sequences ===============
    print("\n=== Evaluating Summarized Task Sequences ===")
    summarized_tasks = load_jsonl("summarized_task_sequences.jsonl")
    
    # Try to load existing results
    existing_summarized_results = load_results_from_jsonl(summarized_results_jsonl)
    if existing_summarized_results:
        summarized_results = existing_summarized_results
        print(f"Loaded existing summarized results from {summarized_results_jsonl}")
    else:
        summarized_results = {model: [None] * len(summarized_tasks) for model in models_to_evaluate}

    for model in models_to_evaluate:
        # Ensure the model exists in results and has the right length
        if model not in summarized_results:
            summarized_results[model] = [None] * len(summarized_tasks)
        # Make sure the list is long enough
        if len(summarized_results[model]) < len(summarized_tasks):
            summarized_results[model].extend([None] * (len(summarized_tasks) - len(summarized_results[model])))
            
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            for idx, task in enumerate(summarized_tasks):
                # Skip tasks that have already been evaluated
                if idx < len(summarized_results[model]) and summarized_results[model][idx] is not None:
                    continue
                futures[executor.submit(evaluate_task_with_trajectory, task, model)] = idx
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Summarized tasks with {model}"):
                idx = futures[future]
                try:
                    success = future.result()
                    summarized_results[model][idx] = success
                    
                    # Append this result to the JSONL file
                    result_entry = {
                        "model": model,
                        "task_idx": idx,
                        "success": success
                    }
                    save_result_to_jsonl(result_entry, summarized_results_jsonl)
                    
                except Exception as e:
                    summarized_results[model][idx] = False
                    print(f"Error processing task {idx} with model {model}: {e}")
                    
                    # Also log the failure
                    result_entry = {
                        "model": model,
                        "task_idx": idx,
                        "success": False,
                        "error": str(e)
                    }
                    save_result_to_jsonl(result_entry, summarized_results_jsonl)
    
    # Save the final consolidated results
    save_final_results(summarized_results, summarized_results_file)
    
    # ========== Evaluate Sequence Task Sequences ===============
    print("\n=== Evaluating Sequence Task Sequences ===")
    sequence_tasks = load_jsonl("task_sequences.jsonl")
    sequence_groups = group_tasks_by_sequence(sequence_tasks)
    
    # Try to load existing results
    existing_sequence_results = load_results_from_jsonl(sequence_results_jsonl)
    if existing_sequence_results:
        sequence_results = existing_sequence_results
        print(f"Loaded existing sequence results from {sequence_results_jsonl}")
    else:
        sequence_results = {model: [None] * len(sequence_groups) for model in models_to_evaluate}
    
    for model in models_to_evaluate:
        # Ensure the model exists in results and has the right length
        if model not in sequence_results:
            sequence_results[model] = [None] * len(sequence_groups)
        # Make sure the list is long enough
        if len(sequence_results[model]) < len(sequence_groups):
            sequence_results[model].extend([None] * (len(sequence_groups) - len(sequence_results[model])))
            
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            for idx, sequence in enumerate(sequence_groups):
                # Skip sequences that have already been evaluated
                if idx < len(sequence_results[model]) and sequence_results[model][idx] is not None:
                    continue
                futures[executor.submit(evaluate_sequence_with_trajectory, sequence, model)] = idx
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Sequence tasks with {model}"):
                idx = futures[future]
                try:
                    success_list = future.result()
                    sequence_results[model][idx] = success_list
                    
                    # Append this result to the JSONL file
                    result_entry = {
                        "model": model,
                        "task_idx": idx,
                        "success": success_list
                    }
                    save_result_to_jsonl(result_entry, sequence_results_jsonl)
                    
                except Exception as e:
                    sequence_results[model][idx] = [False] * len(sequence_groups[idx])
                    print(f"Error processing sequence {idx} with model {model}: {e}")
                    
                    # Also log the failure
                    result_entry = {
                        "model": model,
                        "task_idx": idx,
                        "success": [False] * len(sequence_groups[idx]),
                        "error": str(e)
                    }
                    save_result_to_jsonl(result_entry, sequence_results_jsonl)

    # Save final consolidated results
    save_final_results(sequence_results, sequence_results_file)

    # ========== Baseline Evaluation on ~200 Tasks from Original Dataset ===============
    print("\n=== Baseline Evaluation on ~200 Tasks from Original Dataset ===")
    all_tasks = load_jsonl("task_sequences.jsonl")  # Using the original dataset (task_sequences.jsonl)
    if len(all_tasks) > 200:
        # Use a fixed seed for reproducibility
        random.seed(42)
        baseline_tasks = random.sample(all_tasks, 200)
    else:
        baseline_tasks = all_tasks
    
    # Try to load existing results
    existing_baseline_results = load_results_from_jsonl(baseline_results_jsonl)
    if existing_baseline_results:
        baseline_results = existing_baseline_results
        print(f"Loaded existing baseline results from {baseline_results_jsonl}")
    else:
        baseline_results = {model: [None] * len(baseline_tasks) for model in models_to_evaluate}
    
    for model in models_to_evaluate:
        # Ensure the model exists in results and has the right length
        if model not in baseline_results:
            baseline_results[model] = [None] * len(baseline_tasks)
        # Make sure the list is long enough
        if len(baseline_results[model]) < len(baseline_tasks):
            baseline_results[model].extend([None] * (len(baseline_tasks) - len(baseline_results[model])))
            
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            for idx, task in enumerate(baseline_tasks):
                # Skip tasks that have already been evaluated
                if idx < len(baseline_results[model]) and baseline_results[model][idx] is not None:
                    continue
                futures[executor.submit(evaluate_task_with_trajectory, task, model)] = idx
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Baseline tasks with {model}"):
                idx = futures[future]
                try:
                    success = future.result()
                    baseline_results[model][idx] = success
                    
                    # Append this result to the JSONL file
                    result_entry = {
                        "model": model,
                        "task_idx": idx,
                        "success": success
                    }
                    save_result_to_jsonl(result_entry, baseline_results_jsonl)
                    
                except Exception as e:
                    baseline_results[model][idx] = False
                    print(f"Error processing task {idx} with model {model}: {e}")
                    
                    # Also log the failure
                    result_entry = {
                        "model": model,
                        "task_idx": idx,
                        "success": False,
                        "error": str(e)
                    }
                    save_result_to_jsonl(result_entry, baseline_results_jsonl)

    # Save final consolidated results
    save_final_results(baseline_results, baseline_results_file)
    
    # Print final results
    print("\nSummarized Task Sequences Evaluation Results:")
    for model in models_to_evaluate:
        success_count = sum(1 for s in summarized_results[model] if s)
        total = len(summarized_results[model])
        print(f"Model: {model}, Successful: {success_count}/{total}")

    print("\nTask Sequences Evaluation Results:")
    for model, seqs in sequence_results.items():
        print(f"\nModel: {model}")
        for i, seq in enumerate(seqs):
            success_count = sum(1 for s in seq if s)
            print(f"  Sequence {i+1}: {seq} (Successful tasks: {success_count}/{len(seq)})")

    print("\nBaseline Evaluation Results:")
    for model in models_to_evaluate:
        success_count = sum(1 for s in baseline_results[model] if s)
        total = len(baseline_results[model])
        print(f"Model: {model}, Successful: {success_count}/{total}")


if __name__ == "__main__":
    main() 