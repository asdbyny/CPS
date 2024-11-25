import re
import os
import csv
import json
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch


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


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Classify Incomplete Problem")

    # Add arguments
    parser.add_argument('--model', type=str, default='Llama-3-70B', help='Model to use (default: Llama-3-70B)')

    # Parse the arguments
    args = parser.parse_args()
    
    # Print all the arguments
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    if args.model == "deepseek-math-7b-rl":
        model_name = "deepseek-ai/deepseek-math-7b-rl"
    elif args.model == "Llama-3-70B":
        model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
    elif args.model == "internlm2-math-20b":
        model_name = "internlm/internlm2-math-20b"
    elif args.model == "Qwen1.5-72B-Chat":
        model_name = "Qwen/Qwen1.5-72B-Chat"
    elif args.model == "Mixtral-8x22B-Instruct-v0.1":
        model_name = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    elif args.model == "Yi-1.5-34B-Chat":
        model_name = "01-ai/Yi-1.5-34B-Chat"

    if args.model == "internlm2-math-20b":
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = model.eval()
    elif args.model == "Yi-1.5-34B-Chat":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif args.model == "test":
        pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # Set the base directory for the data
    base_directory = "../subset_final/"

    # List and sort competition directories under the base directory
    competition_dirs = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    competition_dirs.sort(key=numeric_sort_key)

    # Prepare CSV file for output
    output_csv_file = f"../output/inference_{args.model}_greedy.csv"
    fieldnames = ['Competition', 'ID', 'K', 'Problem', 'Prompt', 'Response']

    skip = False
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
                    if args.model in ["deepseek-math-7b-rl", "Mixtral-8x22B-Instruct-v0.1", "Yi-1.5-34B-Chat"]:
                        messages = [
                            {"role": "user", "content": prompt},
                        ]
                    elif args.model == "Qwen1.5-72B-Chat":
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ]
                    else:
                        messages = [
                            {"role": "system", "content": ""},
                            {"role": "user", "content": prompt},
                        ]


                    # decoding setting
                    if args.model == "internlm2-math-20b":
                        response, history = model.chat(tokenizer, prompt, do_sample=False, num_beams=1, temperature=None, top_p=None, history=[], meta_instruction="", max_new_tokens=1024)

                    elif args.model == "Qwen1.5-72B-Chat":
                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                        generated_ids = model.generate(
                            model_inputs.input_ids,
                            max_new_tokens=1024,
                            do_sample=False,
                            num_beams=1,
                            temperature=None,
                            top_p=None,
                        )
                        generated_ids = [
                            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                        ]

                        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    else:
                        if args.model == "Yi-1.5-34B-Chat":
                            input_ids = tokenizer.apply_chat_template(
                                conversation=messages, 
                                tokenize=True, 
                                return_tensors='pt'
                            ).to(model.device)
                        else:
                            input_ids = tokenizer.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                return_tensors="pt"
                            ).to(model.device)

                        if args.model in ["Llama-3-80B"]:
                            terminators = [
                                tokenizer.eos_token_id,
                                tokenizer.convert_tokens_to_ids("<|eot_id|>")
                            ]

                            outputs = model.generate(
                                input_ids,
                                max_new_tokens=1024,
                                eos_token_id=terminators,
                                do_sample=False,
                                num_beams=1,
                                temperature=None,
                                top_p=None,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        elif args.model in ["deepseek-math-7b-rl"]:
                            model.generation_config = GenerationConfig.from_pretrained(model_name)
                            model.generation_config.pad_token_id = model.generation_config.eos_token_id

                            outputs = model.generate(
                                input_ids,
                                max_new_tokens=1024,
                                do_sample=False,
                                num_beams=1,
                                temperature=None,
                                top_p=None,
                            )
                        else:
                            outputs = model.generate(
                                input_ids,
                                max_new_tokens=1024,
                                do_sample=False,
                                num_beams=1,
                                temperature=None,
                                top_p=None,
                                pad_token_id=tokenizer.eos_token_id,
                            )

                        response = outputs[0][input_ids.shape[-1]:]
                        response = tokenizer.decode(response, skip_special_tokens=True)

                    # Write to CSV file
                    writer.writerow({'Competition': os.path.basename(competition), 'ID': i, 'K': k+1, 'Problem': problem, 'Prompt': prompt, 'Response': response})
                    csvfile.flush()  # Flush data to disk after each write

                i = i+1


if __name__ == "__main__":
    main()