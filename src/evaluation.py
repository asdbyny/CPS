import argparse
import logging
import os
from datetime import datetime

from tqdm import tqdm

from config import config
from logger import setup_logger
from models import ModelWrapper
from prompts import (load_coarse_grained_novelty_evaluation_prompt,
                     load_correctness_evaluation_prompt,
                     load_fine_grained_novelty_evaluation_prompt)
from utils import extract_yes_no, load_json, save_json

save_interval = config["experiment"][
    "save_interval"
]  # Save results after every 20 samples
evaluators = ["claude-3-opus", "gemini-1.5-pro", "gpt-4"]


def main():
    parser = argparse.ArgumentParser(description="Run the evaluation program.")
    parser.add_argument(
        "--model_to_evaluate",
        type=str,
        default="Deepseek-math-7b-rl",
        help="The model was used in the experiment and will be evaluated.",
    )
    args = parser.parse_args()
    model_to_evaluate = args.model_to_evaluate

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        config["logging"]["log_dir"],
        f"generation_{model_to_evaluate}_{timestamp}.log",
    )
    logger = setup_logger(log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting the evaluation program...")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")
    logger.warning(
        "Ensure all transition sentences and justifications explaining the uniqueness of new solutions are "
        "manually removed to avoid influencing evaluator judgment.\n"
        "These sentences are usually at the beginning or ending of the response."
    )

    data_path = config["file_paths"]["dataset"]
    generation_path = os.path.join(
        config["file_paths"]["generation"], f"{model_to_evaluate}.json"
    )
    evaluation_dir = config["file_paths"]["evaluation"]
    evaluation_path = os.path.join(evaluation_dir, f"{model_to_evaluate}.json")

    data = load_json(data_path)

    # Evaluation file exists. Continuing unfinished evaluation.
    if os.path.exists(evaluation_path):
        results = load_json(evaluation_path)
    # Create the evaluation file and copy the experiment results.
    else:
        os.makedirs(evaluation_dir, exist_ok=True)
        results = load_json(generation_path)
        for sample_id, sample in enumerate(results):
            results[sample_id]["correctness"] = {}
            results[sample_id]["coarse_grained_novelty"] = {}
            results[sample_id]["fine_grained_novelty"] = {}

    # Stage 1: Correctness Evaluation
    for model_name in evaluators:
        model = ModelWrapper(model_name)

        for sample_id, sample in tqdm(enumerate(results)):
            # Skip if the evaluation result exists
            if model_name in sample["correctness"]:
                continue

            # Load problem and all solutions
            problem_id = sample["problem_id"]
            problem = data[problem_id]["problem"]
            solutions = list(data[problem_id]["solutions"].values())

            # Load the generated new solution
            new_solution = sample["response"]

            prompt = load_correctness_evaluation_prompt(
                problem, solutions, new_solution
            )
            response = model.generate_response(prompt)
            decision = extract_yes_no(response)  # Return either "YES" or "NO"
            sample["correctness"][model_name] = decision
            results[sample_id] = sample

            # Save every 20 samples
            if sample_id % save_interval == 0:
                save_json(results, evaluation_path)
        save_json(results, evaluation_path)

    # A new solution is classified as correct if all three evalution results are "YES"
    for sample_id, sample in enumerate(results):
        all_yes = all(value == "YES" for value in sample["correctness"].values())
        sample["correctness"]["final_decision"] = "YES" if all_yes else "NO"
        results[sample_id] = sample
    save_json(results, evaluation_path)

    # Stage 2: Coarse-Grained Novelty Assessment
    for model_name in evaluators:
        model = ModelWrapper(model_name)

        for sample_id, sample in tqdm(enumerate(results)):
            # Skip if the evaluation result exists
            if model_name in sample["coarse_grained_novelty"]:
                continue

            # Only correct solution will be evaluated.
            # Otherwise classify decision as "NO" directly.
            if sample["correctness"]["final_decision"] == "NO":
                sample["coarse_grained_novelty"][model_name] = "NO"
                results[sample_id] = sample
                continue

            # Load problem and all solutions
            problem_id = sample["problem_id"]
            problem = data[problem_id]["problem"]
            solutions = list(data[problem_id]["solutions"].values())

            # Load the generated new solution
            new_solution = sample["response"]

            k = sample["k"]
            prompt = load_coarse_grained_novelty_evaluation_prompt(
                problem, solutions, k, new_solution
            )
            response = model.generate_response(prompt)
            decision = extract_yes_no(response)  # Return either "YES" or "NO"
            sample["coarse_grained_novelty"][model_name] = decision
            results[sample_id] = sample

            # Save every 20 samples
            if sample_id % save_interval == 0:
                save_json(results, evaluation_path)
        save_json(results, evaluation_path)

    # Determine the final decision based on majority voting
    for sample_id, sample in enumerate(results):
        yes_count = sum(
            1 for value in sample["coarse_grained_novelty"].values() if value == "YES"
        )
        no_count = sum(
            1 for value in sample["coarse_grained_novelty"].values() if value == "NO"
        )
        sample["coarse_grained_novelty"]["final_decision"] = (
            "YES" if yes_count > no_count else "NO"
        )
        results[sample_id] = sample
    save_json(results, evaluation_path)

    # Stage 3: Fine-Grained Novelty Assessment
    for model_name in evaluators:
        model = ModelWrapper(model_name)

        for sample_id, sample in tqdm(enumerate(results)):
            # Skip if the evaluation result exists
            if model_name in sample["fine_grained_novelty"]:
                continue

            # Only solutions that pass the fine-grained novelty assessment will be evaluated.
            # Only samples where k < n will be evaluated.
            # Otherwise, classify the decision as "NO" directly.
            if (sample["coarse_grained_novelty"]["final_decision"] == "NO") or (
                sample["k"] == sample["n"]
            ):
                sample["fine_grained_novelty"][model_name] = "NO"
                results[sample_id] = sample
                continue

            # Load problem and all solutions
            problem_id = sample["problem_id"]
            problem = data[problem_id]["problem"]
            solutions = list(data[problem_id]["solutions"].values())

            # Load the generated new solution
            new_solution = sample["response"]

            k = sample["k"]
            prompt = load_fine_grained_novelty_evaluation_prompt(
                problem, solutions, k, new_solution
            )
            response = model.generate_response(prompt)
            decision = extract_yes_no(response)  # Return either "YES" or "NO"
            sample["fine_grained_novelty"][model_name] = decision
            results[sample_id] = sample

            # Save every 20 samples
            if sample_id % save_interval == 0:
                save_json(results, evaluation_path)
        save_json(results, evaluation_path)

    # Determine the final decision based on majority voting
    for sample_id, sample in enumerate(results):
        yes_count = sum(
            1 for value in sample["fine_grained_novelty"].values() if value == "YES"
        )
        no_count = sum(
            1 for value in sample["fine_grained_novelty"].values() if value == "NO"
        )
        sample["fine_grained_novelty"]["final_decision"] = (
            "YES" if yes_count > no_count else "NO"
        )
        results[sample_id] = sample
    save_json(results, evaluation_path)

    # Calculate accuarcy
    N = len(results)
    correctness_count = 0
    coarse_grained_novelty_count = 0
    fine_grained_novelty_count = 0

    for sample in results:
        if sample["correctness"]["final_decision"] == "YES":
            correctness_count += 1
        if sample["coarse_grained_novelty"]["final_decision"] == "YES":
            coarse_grained_novelty_count += 1
        if sample["fine_grained_novelty"]["final_decision"] == "YES":
            fine_grained_novelty_count += 1

    correctness_ratio = correctness_count / N
    novelty_ratio = coarse_grained_novelty_count / N
    novel_unknown_ratio = fine_grained_novelty_count / N
    if correctness_count != 0:
        novelty_to_correctness_ratio = coarse_grained_novelty_count / correctness_count
    else:
        novelty_to_correctness_ratio = 0
    if coarse_grained_novelty_count != 0:
        novel_unknown_to_novelty_ratio = (
            fine_grained_novelty_count / coarse_grained_novelty_count
        )
    else:
        novel_unknown_to_novelty_ratio = 0

    logger.info(f"The evaluation result for {model_to_evaluate} is as follows:")
    logger.info(f"Correctness Ratio: {correctness_ratio:.2%}")
    logger.info(f"Novelty Ratio: {novelty_ratio:.2%}")
    logger.info(f"Novel-Unknown Ratio: {novel_unknown_ratio:.2%}")
    logger.info(f"Novelty-to-Correctness Ratio: {novelty_to_correctness_ratio:.2%}")
    logger.info(f"Novel-Unknown-to-Novelty Ratio: {novel_unknown_to_novelty_ratio:.2%}")


if __name__ == "__main__":
    main()
