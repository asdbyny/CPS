import re
import os
import json
import csv
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# Helper function to extract and sort based on numeric values in file or directory names
def numeric_sort_key(name):
    numbers = re.findall(r'\d+', name)
    return int(numbers[0]) if numbers else float('inf')  # Handle cases with no numbers reasonably

def classify_math_problem(problem, solution):
    # Generate the prompt
    sys_prompt = f"Please identify the main concept involved in the problem. Classify the following math problem into one of the following categories: arithmetic, algebra, counting, geometry, number theory, probability, or other. Provide a brief explanation of your choice."
    user_prompt = f"**Problem:**\n{problem}\n\n**Solution:**\n{solution}\n\n**Instructions:**\n1. Identify the main concept or concepts involved in the problem.\n2. Choose the category that best represents the main concept(s).\n3. Explain why you chose the selected category.\n\n**Expected Output:**\nConcept: []\nCategory: []\nSimple Explanation: []"

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


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Classify Math Category")

    # Add arguments
    parser.add_argument('--model', type=str, default='Llama-3-70B', help='Model to use (default: Llama-3-70B)')

    # Parse the arguments
    args = parser.parse_args()

    # Print all the arguments
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
        
    if args.model == "Llama-3-8B":
        model = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif args.model == "Llama-3-70B":
        model = "meta-llama/Meta-Llama-3-70B-Instruct"

    # # Directory containing JSON files
    # competition = "AHSME"
    # directory = f"../data/{competition}/1950_AHSME_Problems/"

    # Set the base directory for the data
    base_directory = "../data/"

    # List and sort competition directories under the base directory
    competition_dirs = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    competition_dirs.sort(key=numeric_sort_key)

    # Prepare CSV file for output
    output_csv_file = f"../data/Math_Category_Classification_{args.model}_greedy.csv"
    fieldnames = ['Competition ID', 'Problem ID', 'Response']

    skip = True
    with open(output_csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:  # Check if file is empty to write header
            writer.writeheader()

        # Process each competition directory
        for competition in competition_dirs:
            print(f"Processing competition: {competition}")
            # Navigate into each competition's directory and sort subdirectories/files
            subdirs = [os.path.join(competition, d) for d in os.listdir(competition) if os.path.isdir(os.path.join(competition, d))]
            # subdirs.sort(key=numeric_sort_key)
            subdirs.sort()
            
            # Process each subdirectory within the competition
            for subdir in subdirs:
                print(f"Processing directory: {subdir}")
                # List and sort all JSON files within this subdirectory
                json_files = [f for f in os.listdir(subdir) if f.endswith('.json')]
                json_files.sort(key=numeric_sort_key)  # Sort files numerically based on the first number in the filename
                
                # Process each file
                for file_name in json_files:
                    file_path = os.path.join(subdir, file_name)
                    print(f"Processing file: {file_path}")

                    if file_path == "../data/AMC_8/2004_AMC_8_Problems/Problem_24.json":
                        skip = False
                    if skip:
                        continue
                        
                    competition_id, problem_id, problem, solution = read_problem_and_first_solution(file_path)
                    # print("Competition ID:", competition_id)
                    # print("Problem ID:", problem_id)
                    # print("Problem:", problem)
                    # print("Solution:", solution)

                    # Classify math problem category:
                    sys_prompt, user_prompt = classify_math_problem(problem, solution)

                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ]

                    input_ids = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(model.device)

                    terminators = [
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

                    # outputs = model.generate(
                    #     input_ids,
                    #     max_new_tokens=4096,
                    #     eos_token_id=terminators,
                    #     do_sample=True,
                    #     temperature=0.6,
                    #     top_p=0.9,
                    #     pad_token_id=tokenizer.eos_token_id,
                    # )

                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=4096,
                        eos_token_id=terminators,
                        do_sample=False,
                        num_beams=1,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                    response = outputs[0][input_ids.shape[-1]:]
                    response = tokenizer.decode(response, skip_special_tokens=True)
                    # print("Response:", response)
                    # print("\n")

                    # Write to CSV file
                    writer.writerow({'Competition ID': competition_id, 'Problem ID': problem_id, 'Response': response})
                    csvfile.flush()  # Flush data to disk after each write


if __name__ == "__main__":
    main()