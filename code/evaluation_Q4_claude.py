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

def main():
    # Prepare CSV file for output
    csv_file = f"../Q4/output.csv"

    # Read the existing CSV file
    df = pd.read_csv(csv_file)
    prompts = df["Prompt"]
    df["Novelty_claude"] = ""

    for i, prompt in prompts.items():
        if pd.notna(df.at[i, 'Novelty_claude']) and df.at[i, 'Novelty_claude'] != "":
            continue
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

        df.at[i, 'Novelty_claude'] = response
        df.to_csv(csv_file, index=False)
                

if __name__ == "__main__":
    main()
