import re
import os
import sys
import csv
import time
import json
import argparse

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted


# Helper function to extract and sort based on numeric values in file or directory names
def numeric_sort_key(name):
    numbers = re.findall(r'\d+', name)
    return int(numbers[0]) if numbers else float('inf')  # Handle cases with no numbers reasonably

def classify_incomplete_problem(problem):
    # Generate the prompt
    sys_prompt = f"Please identify if the given math problem is an incomplete math problem. Output YES or NO."
    user_prompt = f"**Problem:**\n{problem}\n\n**Instructions:**\nJust output YES or NO."

    return sys_prompt, user_prompt 

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

def get_novel_solution_prompt(problem, solutions, k):
    k_solutions = solutions[:k]
    solution_prompt = ""
    for i, solution in enumerate(k_solutions):
        solution_prompt += f"Solution {i+1}:\n{solution}\n\n"
    
    prompt = f"""Criteria for evaluating the difference between two mathematical solutions include:
i). If the methods used to arrive at the solutions are fundamentally different, such as algebraic manipulation versus geometric reasoning, they can be considered distinct;
ii). Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the solutions can be considered different;
iii). If two solutions rely on different assumptions or conditions, they are likely to be distinct;
iv). A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;
v). If one solution is significantly simpler or more complex than the other, they can be regarded as essentially different, even if they lead to the same result.

Given the following mathematical problem:
{problem}

And some typical solutions:
{solution_prompt}Please output a novel solution distinct from the given ones for this math problem."""
    
    return prompt


genai.configure(api_key='your_api_key')


def main():
    # Set the base directory for the data
    base_directory = "../subset_final/"

    # List and sort competition directories under the base directory
    competition_dirs = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    competition_dirs.sort(key=numeric_sort_key)

    # Prepare CSV file for output
    output_csv_file = f"../output/inference_gemini-1.5-pro_greedy.csv"
    fieldnames = ['ID', 'K', 'Problem', 'Prompt', 'Response']

    index = 0
    with open(output_csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:  # Check if file is empty to write header
            writer.writeheader()

        # Process each competition directory
        for competition in competition_dirs:
            if competition.endswith(".ipynb_checkpoints"):
                continue

            print(f"Processing competition: {competition}")
            # List and sort all JSON files within this subdirectory
            json_files = [f for f in os.listdir(competition) if f.endswith('.json')]
            json_files.sort(key=numeric_sort_key)  # Sort files numerically based on the first number in the filename
            
            # Process each file
            i = 1
            for file_name in json_files:
                file_path = os.path.join(competition, file_name)

                print(f"Processing file: {file_path}")

                with open(file_path, 'r') as file:
                    data = json.load(file)
            
                problem = data["problem"]
                solutions = list(data["solutions"].values())
            
                for k in range(len(solutions)):
                    prompt = get_novel_solution_prompt(problem, solutions, k+1)
                    custom_id = f"{i}_{k+1}"

                    # prompt style
                    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

                    generation_config = genai.GenerationConfig(temperature=0.0, max_output_tokens=1024)

                    max_retries = 10  # Maximum number of retries
                    retry_count = 0
                    while retry_count < max_retries:
                        try:
                            # Generate content using the configuration
                            response = model.generate_content(prompt, generation_config=generation_config).text
                            time.sleep(1)
                            break
                        except ResourceExhausted as e:
                            print(f"Resource exhausted. Retrying in 5 seconds... (Attempt {retry_count + 1}/{max_retries})")
                            time.sleep(5)
                            retry_count += 1

                    if retry_count == max_retries:
                        print(f"Max retries reached. Could not complete the request.")
                        sys.exit(1)  # Stop the program

                    # Write to CSV file
                    writer.writerow({'ID': i, 'K': k+1, 'Problem': problem, 'Prompt': prompt, 'Response': response})
                    csvfile.flush()  # Flush data to disk after each write

                i = i+1



if __name__ == "__main__":
    main()