import re
import os
import sys
import csv
import time
import json
import argparse
import numpy as np
import pandas as pd

from openai import OpenAI

client = OpenAI()

def main():
    # Prepare CSV file for output
    csv_file = f"../Q4/output.csv"

    # Read the existing CSV file
    df = pd.read_csv(csv_file)
    prompts = df["Prompt"]
    df["Novelty_gpt4"] = ""

    for i, prompt in prompts.items():
        if pd.notna(df.at[i, 'Novelty_gpt4']) and df.at[i, 'Novelty_gpt4'] != "":
            continue
        response = client.chat.completions.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0,
            stream=False
        )
        response = response.choices[0].message.content

        df.at[i, 'Novelty_gpt4'] = response
        df.to_csv(csv_file, index=False)
                

if __name__ == "__main__":
    main()
