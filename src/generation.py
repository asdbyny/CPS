import argparse
import json
import logging
import os
from datetime import datetime

from tqdm import tqdm

from config import config
from logger import setup_logger
from models import ModelWrapper
from prompts import load_novel_solution_generation_prompt

save_interval = config["experiment"][
    "save_interval"
]  # Save results after every 20 samples


def main():
    parser = argparse.ArgumentParser(
        description="Run the novel solution generation program."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Deepseek-math-7b-rl",
        help="The model used to generate novel solutions.",
    )
    args = parser.parse_args()
    model_name = args.model_name

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        config["logging"]["log_dir"],
        f"generation_{model_name}_{timestamp}.log",
    )
    logger = setup_logger(log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting the novel solution generation program...")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")

    model = ModelWrapper(model_name)

    data_path = config["file_paths"]["dataset"]
    with open(data_path, "r") as file:
        data = json.load(file)

    results = []
    for problem_id, sample in tqdm(enumerate(data)):
        problem = sample["problem"]
        solutions = list(sample["solutions"].values())  # All solutinos
        n = len(solutions)  # Total number of solutions

        # k: number of the reference solutions provided in the prompt
        # Interate k from 1, 2, until n
        for k in range(1, n + 1):
            prompt = load_novel_solution_generation_prompt(problem, solutions, k)
            response = model.generate_response(prompt)
            results.append(
                {"problem_id": problem_id, "k": k, "n": n, "response": response}
            )

    output_dir = config["file_paths"]["generation"]
    output_file = os.path.join(output_dir, f"{model_name}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as file:
        json.dump(results, file, indent=4)
    logger.info(f"Results saved to {os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()
