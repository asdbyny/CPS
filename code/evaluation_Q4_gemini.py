import re
import os
import sys
import csv
import time
import json
import argparse
import numpy as np
import pandas as pd

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

genai.configure(api_key='your_api_key')


def main():
    # Prepare CSV file for output
    csv_file = f"../Q4/output.csv"

    # Read the existing CSV file
    df = pd.read_csv(csv_file)
    prompts = df["Prompt"]
    df["Novelty_gemini"] = ""

    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
    generation_config = genai.GenerationConfig(temperature=0.0, max_output_tokens=1024)
    
    for i, prompt in prompts.items():
        if pd.notna(df.at[i, 'Novelty_gemini']) and df.at[i, 'Novelty_gemini'] != "":
            continue
        max_retries = 100  # Maximum number of retries
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Generate content using the configuration
                response = model.generate_content(prompt, generation_config=generation_config).text
                df.at[i, 'Novelty_gemini'] = response
                df.to_csv(csv_file, index=False)
                time.sleep(1)
                break
            except ResourceExhausted as e:
                print(f"Resource exhausted. Retrying in 5 seconds... (Attempt {retry_count + 1}/{max_retries})")
                time.sleep(5)
                retry_count += 1

        if retry_count == max_retries:
            print(f"Max retries reached. Could not complete the request.")
            sys.exit(1)  # Stop the program

if __name__ == "__main__":
    main()
