import json
import time
import argparse
import requests
import os
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import necessary functions from combined_task_generation.py
from combined_task_generation import ensure_container_running, call_gpt_4o

# System prompt for the judge
SYS_JUDGE_PROMPT = """
You are a helpful assistant that evaluates whether a sequence of actions successfully completes a given task.

You will be given:
1. A task description
2. A sequence of actions that were taken
3. The webpage states after each action was executed

Your job is to determine if the actions successfully completed the task.

Output your evaluation as a JSON object with the following fields:
- completed: A boolean indicating whether the task was successfully completed (true/false)
- reasoning: A detailed explanation of your reasoning
- confidence: A number between 0 and 1 indicating your confidence in your evaluation

Assume that parts of the task not visible in the webpage state were completed. For instance, if the task
was to download a file, and the webpage is the download page, then we can assume that the task was completed.
Look at all the webpage states for your determination, not just the final state.
"""

def evaluate_task_sequence(tool, task_data, model="gpt-4o"):
    """
    Evaluate whether a summarized task sequence successfully completes its task.
    
    Args:
        tool: The InSTA tool object to interact with websites
        task_data: Dictionary containing task sequence data
        model: The model to use for evaluation
        
    Returns:
        Dictionary with evaluation results
    """
    website = task_data.get("website")
    task = task_data.get("task")
    action_sequence = task_data.get("action_sequence", [])
    
    if not website or not task or not action_sequence:
        return {
            "website": website,
            "task": task,
            "evaluation": {
                "completed": False,
                "reasoning": "Missing required data (website, task, or action sequence)",
                "confidence": 1.0
            },
            "error": "Missing required data"
        }
    
    try:
        # Load the website
        print(f"Loading website: {website}")
        outputs = tool(url="http://" + website)
        
        # Check if website loaded properly
        if "session ID:" not in outputs:
            return {
                "website": website,
                "task": task,
                "evaluation": {
                    "completed": False,
                    "reasoning": "Could not load website",
                    "confidence": 1.0
                },
                "error": "Website failed to load"
            }
            
        # Extract session ID
        session_id = outputs.split("session ID: `")[1].split("`")[0]
        
        webpage_texts = []
        # Execute each action in the sequence
        for i, action in enumerate(action_sequence):
            print(f"Executing action {i+1}/{len(action_sequence)}: {str(action)[:100]}...")
            if action == "DONE":
                continue
                
            # Convert action to string if it's a dictionary
            action_str = action
            if isinstance(action, dict):
                action_str = json.dumps(action)
            else:
                # Sanitize the string action by trimming whitespace
                action_str = action_str.strip()            
            try:
                outputs = tool(session_id=session_id, action=action_str)
                time.sleep(3)  # Small delay between actions
                
                if "Failed to execute action" in outputs or "The provided action could not be parsed" in outputs:
                    print(f"Action {i+1} failed: {action_str[:100]}...")
                    # return {
                    #     "website": website,
                    #     "task": task,
                    #     "evaluation": {
                    #         "completed": False,
                    #         "reasoning": f"Action {i+1} failed to execute: {action_str[:100]}...",
                    #         "confidence": 0.9
                    #     },
                    #     "error": f"Action execution failed at step {i+1}"
                    # }
                    continue
                
                webpage_text = outputs.split("in markdown:")[1].strip()
                webpage_texts.append(webpage_text[:10000])
            except Exception as e:
                print(f"Error executing action {i+1}: {e}")
                return {
                    "website": website,
                    "task": task,
                    "evaluation": {
                        "completed": False,
                        "reasoning": f"Error executing action {i+1}: {str(e)}",
                        "confidence": 0.9
                    },
                    "error": f"Exception at step {i+1}: {str(e)}"
                }
        
        # Extract final webpage text
        final_webpage_text = ""
        if "in markdown:" in outputs:
            final_webpage_text = outputs.split("in markdown:")[1].strip()
        
        # Evaluate task completion
        user_prompt = f"""
        Task: {task}

        Action Sequence:
        {json.dumps(action_sequence, indent=2)}

        Webpage States:
        {webpage_texts} 

        Based on the above information, did the action sequence successfully complete the task?
        """

        # Call LLM to evaluate
        llm_output = call_gpt_4o(SYS_JUDGE_PROMPT, user_prompt, model=model)
        
        try:
            evaluation = json.loads(llm_output)
            return {
                "website": website,
                "task": task,
                "evaluation": evaluation
            }
        except json.JSONDecodeError:
            # If LLM output isn't valid JSON, try to extract key information
            completed = "true" in llm_output.lower() and "completed" in llm_output.lower()
            return {
                "website": website,
                "task": task,
                "evaluation": {
                    "completed": completed,
                    "reasoning": llm_output,
                    "confidence": 0.5
                },
                "error": "Failed to parse LLM output as JSON"
            }
            
    except Exception as e:
        print(f"Error evaluating task sequence: {e}")
        import traceback
        traceback.print_exc()
        return {
            "website": website,
            "task": task,
            "evaluation": {
                "completed": False,
                "reasoning": f"Evaluation error: {str(e)}",
                "confidence": 0.5
            },
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Evaluate summarized task sequences")
    parser.add_argument("--input_file", type=str, default="summarized_task_sequences.jsonl",
                        help="Input file containing summarized task sequences")
    parser.add_argument("--output_file", type=str, default="evaluated_task_sequences.jsonl",
                        help="Output file for evaluation results")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers for parallel evaluation")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to use for evaluation")
    args = parser.parse_args()
    
    # Ensure InSTA container is running
    if not ensure_container_running():
        print("Failed to start InSTA container. Exiting.")
        return
    
    # Load summarized task sequences
    task_sequences = []
    try:
        with open(args.input_file, 'r') as f:
            for line in f:
                try:
                    task_data = json.loads(line.strip())
                    task_sequences.append(task_data)
                except json.JSONDecodeError:
                    print(f"Error parsing line: {line[:100]}...")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return
    
    task_sequences = task_sequences[-24:]
    print(f"Loaded {len(task_sequences)} task sequences for evaluation")
    
    # Create tool instance
    from insta import InstaTransformersTool
    tool = InstaTransformersTool()
    
    # Evaluate task sequences
    results = []
    
    if args.num_workers > 1:
        # Parallel evaluation
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for task_data in task_sequences:
                future = executor.submit(evaluate_task_sequence, tool, task_data, args.model)
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Write result to output file immediately
                    with open(args.output_file, 'a') as f:
                        f.write(json.dumps(result) + "\n")
                        
                except Exception as e:
                    print(f"Error in task evaluation: {e}")
    else:
        # Sequential evaluation
        for task_data in tqdm(task_sequences, desc="Evaluating"):
            try:
                result = evaluate_task_sequence(tool, task_data, args.model)
                results.append(result)
                
                # Write result to output file immediately
                with open(args.output_file, 'a') as f:
                    f.write(json.dumps(result) + "\n")
                    
            except Exception as e:
                print(f"Error in task evaluation: {e}")
    
    # Summarize results
    completed_count = sum(1 for r in results if r.get("evaluation", {}).get("completed", False))
    total_count = len(results)
    completion_rate = completed_count / total_count if total_count > 0 else 0
    
    print(f"\nEvaluation Summary:")
    print(f"Total task sequences evaluated: {total_count}")
    print(f"Successfully completed tasks: {completed_count}")
    print(f"Completion rate: {completion_rate:.2%}")
    print(f"Results written to: {args.output_file}")

if __name__ == "__main__":
    main()
