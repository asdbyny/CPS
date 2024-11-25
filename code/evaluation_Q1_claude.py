import re
import os
import sys
import csv
import time
import json
import argparse
import numpy as np
import pandas as pd

import anthropic

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="your_api_key",
)

# Helper function to extract and sort based on numeric values in file or directory names
def numeric_sort_key(name):
    numbers = re.findall(r'\d+', name)
    return int(numbers[0]) if numbers else float('inf')  # Handle cases with no numbers reasonably

def read_problem_and_first_solution(filepath):
    # Check if the file exists
    if not os.path.isfile(filepath):
        print("Error: The provided path is not a file.")
        return None
    
    with open(filepath, "r") as file:
        json_data = json.load(file)
        competition_id = json_data["competition_id"]
        problem_id = json_data["problem_id"]
        problem = json_data["problem"]
        
        # Find the first solution
        try:
            first_solution = next(iter(json_data["solutions"].values()))
        except Exception as e:
            print(f"An error occurred when retrieving the first solution: {e}")
            first_solution = "Error"
        
        return competition_id, problem_id, problem, first_solution

def get_correctness_prompt(problem, solutions, new_solution, switch_yes_and_no=False):
    solution_prompt = ""
    n_solutions = len(solutions)
    if n_solutions > 1:
        solution_prompt += f"Solution 1:\n{solutions[0]}\n\n"
        solution_prompt += f"Solution 2:\n{solutions[1]}\n\n"
    else:
        solution_prompt += f"Solution 1:\n{solutions[0]}\n\n"
    
    if switch_yes_and_no:
        prompt = f"""Given the following mathematical problem:
{problem}

Reference solutions:
{solution_prompt}
New solution:
{new_solution}

Please output NO if the new solution leads to the same result as the reference solutions; otherwise, output YES."""

    else:
        prompt = f"""Given the following mathematical problem:
{problem}

Reference solutions:
{solution_prompt}
New solution:
{new_solution}

Please output YES if the new solution leads to the same result as the reference solutions; otherwise, output NO."""
    
    return prompt


def get_novelty_prompt(problem, solutions, k, new_solution, switch_yes_and_no=False):
    k_solutions = solutions[:k]
    solution_prompt = ""
    for i, solution in enumerate(k_solutions):
        solution_prompt += f"Solution {i+1}:\n{solution}\n\n"
    
    if switch_yes_and_no:
        prompt = f"""Criteria for evaluating the novelty of a new mathematical solution include:
1. If the new solution used to arrive at the solutions is fundamentally different from reference solutions, such as algebraic manipulation versus geometric reasoning, it can be considered novel;
2. Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the new solution can be considered novel;
3. If the new solution relies on different assumptions or conditions, it should be considered novel;
4. A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;
5. If the new solution is significantly simpler or more complex than the others, it can be regarded as essentially novel, even if they lead to the same result.

Given the following mathematical problem:
{problem}

Reference solutions:
{solution_prompt}
New solution:
{new_solution}

Please output NO if the new solution is a novel solution; otherwise, output YES."""

    else:
        prompt = f"""Criteria for evaluating the novelty of a new mathematical solution include:
1. If the new solution used to arrive at the solutions is fundamentally different from reference solutions, such as algebraic manipulation versus geometric reasoning, it can be considered novel;
2. Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the new solution can be considered novel;
3. If the new solution relies on different assumptions or conditions, it should be considered novel;
4. A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;
5. If the new solution is significantly simpler or more complex than the others, it can be regarded as essentially novel, even if they lead to the same result.

Given the following mathematical problem:
{problem}

Reference solutions:
{solution_prompt}
New solution:
{new_solution}

Please output YES if the new solution is a novel solution; otherwise, output NO."""
    
    return prompt

def get_fine_novelty_prompt(problem, solutions, k, new_solution):
    if len(solutions) == k:
        return None
    k_solutions = solutions[k:]
        
    solution_prompt = ""
    for i, solution in enumerate(k_solutions):
        solution_prompt += f"Solution {i+1}:\n{solution}\n\n"
    
    prompt = f"""Criteria for evaluating the novelty of a new mathematical solution include:
1. If the new solution used to arrive at the solutions is fundamentally different from reference solutions, such as algebraic manipulation versus geometric reasoning, it can be considered novel;
2. Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the new solution can be considered novel;
3. If the new solution relies on different assumptions or conditions, it should be considered novel;
4. A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;
5. If the new solution is significantly simpler or more complex than the others, it can be regarded as essentially novel, even if they lead to the same result.

Given the following mathematical problem:
{problem}

Reference solutions:
{solution_prompt}
New solution:
{new_solution}

Please output YES if the new solution is a novel solution; otherwise, output NO."""
    
    return prompt


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Creative Math")

    # Add arguments
    parser.add_argument('--model', type=str, default='Llama-3-70B', help='Model to use (default: Llama-3-70B)')
    parser.add_argument('--switch_yes_and_no', action='store_true', help='Switch YES and NO in the prompt (default: False)')

    # Parse the arguments
    args = parser.parse_args()
    
    # Print all the arguments
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    # Set the base directory for the data
    base_directory = "../subset_final/"
    json_base_directory = "../subset_final/"  # Assuming JSON files are in the same base directory

    # List and sort competition directories under the base directory
    competition_dirs = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    competition_dirs.sort(key=numeric_sort_key)

    # Prepare CSV file for output
    output_csv_file = f"../output_final/inference_{args.model}_greedy.csv"

    # Read the existing CSV file
    df = pd.read_csv(output_csv_file)

    prompts = []
    if 'Prompt_Fine_Novelty' not in df.columns:
        df['Prompt_Fine_Novelty'] = np.nan
    if 'Fine_novelty_claude' not in df.columns:
        df['Fine_novelty_claude'] = ''  # Initialize the 'Novelty' column for all rows
        
    # Process each row to add the new column
    for index, row in df.iterrows():
        competition = row['Competition']
        file_id = row['ID']
        k = int(row['K'])
        response = row['New_Solution']

        # Find the corresponding JSON file
        json_file_name = f"{file_id}.json"
        file_path = os.path.join(json_base_directory, competition, json_file_name)
        
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)

            problem = data["problem"]
            solutions = list(data["solutions"].values())
            
            # # correctness
            # prompt = get_correctness_prompt(problem, solutions, response, args.switch_yes_and_no)
            # prompts.append(prompt)

            # # novelty
            # prompt = get_novelty_prompt(problem, solutions, k, response, args.switch_yes_and_no)
            # if row['Correctness_all_yes'] == 'YES':
            #     prompts.append(prompt)
            # else:
            #     prompts.append(np.nan)
            #     df.loc[index, 'Novelty_claude'] = 'N/A'

            # fine novelty
            prompt = get_fine_novelty_prompt(problem, solutions, k, response)
            if row['Novelty_major_voting'] == 'YES' and k < len(solutions):
                prompts.append(prompt)
            else:
                prompts.append(np.nan)
                df.loc[index, 'Fine_novelty_claude'] = 'N/A'

        else:
            print(f"JSON file not found for {file_path}")
            prompts.append("Error: JSON file not found")
        

    prompts = pd.Series(prompts)

    # # correctness
    # df['Prompt_Correctness'] = prompts

    # # novelty
    # df['Prompt_Novelty'] = prompts

    # fine novelty
    df['Prompt_Fine_Novelty'] = prompts
    
    for i, prompt in prompts.items():
        # if pd.notna(df.at[i, 'Correctness']) and df.at[i, 'Correctness'] != '':
        #     continue
        # if pd.notna(df.at[i, 'Novelty_claude']) and df.at[i, 'Novelty_claude'] != '':
        #     continue
        if pd.notna(df.at[i, 'Fine_novelty_claude']) and df.at[i, 'Fine_novelty_claude'] != '':
            continue
        # prompt style
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            temperature=0.0,
            system="",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response = message.content[0].text
        # df.at[i, 'Correctness'] = response
        # df.at[i, 'Novelty_claude'] = response
        df.at[i, 'Fine_novelty_claude'] = response
        df.to_csv(output_csv_file, index=False)

if __name__ == "__main__":
    main()
