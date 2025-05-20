import os
import re
import json
from ollama import Client

def slugify(text: str) -> str:
    """
    Transforme une chaîne en slug pour nom de fichier :
    remplace les caractères non alphanumériques par des underscores.
    """
    return re.sub(r"[^\w]+", "_", text).strip("_")

def generate_fr_base_questions(
    topic: str,
    num_questions: int,
    output_dir: str = "question_groups",
    model_name: str = "llama3.2:latest",
    use_gpu: bool = True,
    max_attempts: int = 3
):
    """
    Génère un fichier JSONL de questions simples en français sur un thème donné,
    une question à la fois, et le place dans `output_dir`.
    Affiche chaque question dès qu'elle est générée.
    """
    os.makedirs(output_dir, exist_ok=True)
    slug = slugify(topic)
    output_path = os.path.join(output_dir, f"{slug}.jsonl")
    if os.path.exists(output_path):
        os.remove(output_path)

    client = Client()
    print(f"💻 Client Ollama initialisé avec '{model_name}' (GPU={'oui' if use_gpu else 'non'})\n")

    sys_prompt = (
        "Tu es un assistant expert en génération de questions simples. "
        "Tu ne renvoies **que** la question, sans guillemets superflus, "
        "sans numéros ni préfixes, juste le texte de la question."
    )

    options = {"temperature": 0.7, "top_p": 0.9, "num_predict": 100}
    if use_gpu:
        options["device"] = "cuda"

    with open(output_path, "w", encoding="utf-8") as fout:
        for i in range(1, num_questions + 1):
            user_prompt = (
                f"Génère une question simple en français sur le thème « {topic} » "
                f"(question {i} sur {num_questions})."
            )
            question = None

            for attempt in range(1, max_attempts + 1):
                print(f"  ⚙️ Génération question {i}/{num_questions}, essai {attempt}/{max_attempts}…")
                resp = client.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    options=options
                )
                text = resp.message.content.strip()
                if text:
                    question = text
                    break
                print("    ⚠️ Sortie vide, nouvel essai…")

            if question is None:
                raise RuntimeError(f"🛑 Échec après {max_attempts} tentatives pour la question {i}.")

            print(f"  ✅ Question {i}: {question}")
            fout.write(json.dumps({"question": question}, ensure_ascii=False) + "\n")

    print(f"\n🎉 Terminé ! {num_questions} questions générées dans « {output_path} ».")

def run_generate_questions(
    topic: str,
    num_questions: int,
    output_dir: str = "dataset/question_groups",
    model_name: str = "llama3.2:latest",
    use_gpu: bool = True
):
    """
    Wrapper to call generate_fr_base_questions from other files.
    """
    generate_fr_base_questions(topic, num_questions, output_dir, model_name, use_gpu)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate French simple questions")
    parser.add_argument("topic", help="Le thème des questions")
    parser.add_argument("num_questions", type=int, help="Nombre de questions à générer")
    parser.add_argument("--output_dir", default="dataset/question_groups", help="Répertoire de sortie")
    parser.add_argument("--model_name", default="llama3.2:latest", help="Nom du modèle Ollama")
    parser.add_argument("--no_gpu", action="store_true", help="Ne pas utiliser le GPU")
    args = parser.parse_args()
    run_generate_questions(
        topic=args.topic,
        num_questions=args.num_questions,
        output_dir=args.output_dir,
        model_name=args.model_name,
        use_gpu=not args.no_gpu,
    )
