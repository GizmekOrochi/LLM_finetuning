import os
import json
from fusion_generator import generate_fr_dataset

def load_question_groups(folder_path="dataset/question_groups"):
    groups = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".jsonl"):
            continue
        file_path = os.path.join(folder_path, fname)
        questions = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if isinstance(data, dict) and "question" in data:
                    questions.append(data["question"])
                elif isinstance(data, str):
                    questions.append(data)
                else:
                    questions.append(data)
        groups.append(questions)
    return groups


def run_fusion_pipeline(
    question_folder: str = "dataset/question_groups",
    temp_folder: str = "dataset/fusion_datasets",
    merged_output_path: str = "dataset/batch_fusion_generator/batch_fusion_dataset.jsonl"
):

    os.makedirs(temp_folder, exist_ok=True)
    question_groups = load_question_groups(question_folder)
    temp_paths = []

    for idx, questions in enumerate(question_groups, start=1):
        temp_path = os.path.join(temp_folder, f"small_dataset_{idx}.jsonl")
        print(f"Génération du dataset #{idx} → {temp_path}")
        generate_fr_dataset(
            questions=questions,
            output_path=temp_path
        )
        temp_paths.append(temp_path)

    merged_entries = []
    for path in temp_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    merged_entries.append(json.loads(line))

    os.makedirs(os.path.dirname(merged_output_path), exist_ok=True)
    print(f"Enregistrement du dataset fusionné → {merged_output_path}")

    with open(merged_output_path, "w", encoding="utf-8") as fout:
        for entry in merged_entries:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"🎉 Terminé : {len(merged_entries)} entrées fusionnées dans '{merged_output_path}'.")


if __name__ == '__main__':
    run_fusion_pipeline()
