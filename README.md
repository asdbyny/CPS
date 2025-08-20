# CPS
# Math Problem Evaluation System (CPS Evaluation)

An automated system for evaluating mathematical problem solutions, based on the Creative Process Score (CPS) model. It quantitatively assesses solutions across three dimensions: **correctness**, **novelty**, and **process quality**.


## Project Features

- Automatically loads math problem data (supports `subset.json` format)
- Generates solutions to math problems (simulated or via large language models)
- Evaluates Creative Process Scores (CPS) for solutions, including:
  - Correctness verification (via multi-model review consensus)
  - Novelty score (degree of difference from reference solutions)
  - Process quality score (step validity and redundancy)
- Outputs evaluation results (supports JSON format saving)
