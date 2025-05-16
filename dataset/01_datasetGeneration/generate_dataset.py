"""
generate_dataset.py

Generates a JSONL dataset of 10,000 entries with prompts like:
{"question": "Where is <Name> ?", "answer": "In <Country>!"}

Usage:
    pip install Faker
    python3 generate_dataset.py --entries 10000 --output data.jsonl

"""

import argparse
from faker import Faker
import json
import random

def generate_dataset(num_entries, actions, output_file):
    faker = Faker()
    unique_names = set()
    while len(unique_names) < num_entries:
        unique_names.add(faker.first_name() + " " + faker.last_name())
    names = list(unique_names)

    with open(output_file, 'w', encoding='utf-8') as f:
        for name in names:
            action = random.choice(actions)
            entry = {
                "question": f"What does {name} ?",
                "answer": f"{name} is {action}!"
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSONL dataset for Unsloth fine-tuning")
    parser.add_argument('--entries', type=int, default=10000, help='Number of entries to generate')
    parser.add_argument('--output', type=str, default='data.jsonl', help='Output JSONL file path')
    args = parser.parse_args()

    actions = ["cooking", "napping", "playing", "coding", "running", 
                 "driving", "drinking", "eating", "working", "waiting"]
    generate_dataset(args.entries, actions, args.output)
    print(f"Generated {args.output} with {args.entries} entries.")
