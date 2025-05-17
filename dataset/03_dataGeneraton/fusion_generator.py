import os
import json
from itertools import combinations
from ollama import Client

def generate_fr_dataset(
    questions,
    output_path,
    model_name="llama3.2:latest",
    use_gpu=True,
    max_attempts=10
):
    # S'assurer que le dossier existe
    folder = os.path.dirname(output_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    client = Client()
    print(f"💻 Client Ollama initialisé avec modèle '{model_name}' (GPU={'oui' if use_gpu else 'non'})")

    # Configuration des prompts
    sys_complex = (
        "Tu es un assistant qui **ne renvoie que** la question complexe fusionnant les deux questions,"
        " sans explication, sans balises, sans guillemets superflus,"  
        " sans inventer aucune information."
    )
    sys_rephrase = (
        "Tu es un assistant qui **ne renvoie que** la question reformulée,"
        " sans explication, sans balises, sans guillemets superflus."
    )

    basic_variants = [
        ("simple", sys_complex,
         lambda q1, q2: (
             f"Fusionne ces deux questions en une seule question volontairement très simple:\n"
             f"- {q1}\n"
             f"- {q2}"
         )
        )
    ]

    advanced_variants = [
        ("fluide",     sys_rephrase, lambda q: f"Reformule cette question de manière fluide et claire : « {q} »"),
        ("complexe2",  sys_rephrase, lambda q: f"Reformule cette question de manière volontairement complexe : « {q} »"),
        ("courte",     sys_rephrase, lambda q: f"Reformule cette question en version courte et concise : « {q} »"),
        #("ouverte",    sys_rephrase, lambda q: f"Transforme cette question en question ouverte pour lancer la discussion : « {q} »"),
        #("academique", sys_rephrase, lambda q: f"Rédige cette question dans un style académique et formel : « {q} »"),
        #("fermee",     sys_rephrase, lambda q: f"Reformule en question fermée (oui/non) : « {q} »"),
        #("niveau_ce2", sys_rephrase, lambda q: f"Adapte cette question pour un élève de CE2 : « {q} »"),
        #("suivi",      sys_rephrase, lambda q: f"À partir de « {q} », formule une question de suivi pour approfondir le sujet"),
        #("passive",    sys_rephrase, lambda q: f"Reformule cette question à la voix passive : « {q} »"),
        #("comparative",sys_rephrase, lambda q: f"Transforme cette question pour comparer deux entités : « {q} »")
    ]

    def get_ollama_options():
        opts = {"temperature": 0.2, "top_p": 0.9, "num_predict": 100}
        if use_gpu:
            opts["device"] = "cuda"
        return opts

    def generate_unique_question(system_prompt, user_prompt, seen):
        for attempt in range(1, max_attempts + 1):
            print(f"  ⚙️ Tentative {attempt}/{max_attempts} pour: '{user_prompt[:50]}...'" )
            resp = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                options=get_ollama_options()
            )
            text = resp.message.content.strip()
            if text and text not in seen:
                print(f"    ✅ Obtention: '{text}'")
                return text
            print("    ⚠️ Doublon ou vide, nouvel essai…")
        print(f"    ❌ Échec après {max_attempts} tentatives pour: '{user_prompt}'")
        return None

    dataset = []
    pairs = list(combinations(questions, 2))
    total = len(pairs)
    print(f"🔄 Démarrage de la génération pour {total} paires...")

    count = 0
    for idx, (q1, q2) in enumerate(pairs, start=1):
        print("\n" + "="*60)
        print(f" 🤖 Exemple {idx}/{total}  Réponses générées: {count}")
        print("="*60)
        print(f"Question 1: {q1}\nQuestion 2: {q2}\n")

        seen = set()
        base_answer = f"{q1} {q2}"

        # Variante simple
        name, sys_prompt, tpl = basic_variants[0]
        print(f"🛠️ Génération ({name})…")
        user_prompt = tpl(q1, q2)
        simple_q = generate_unique_question(sys_prompt, user_prompt, seen)
        if not simple_q:
            print(f"⚠️ Échec de la variante simple, passage à l'exemple suivant.")
            continue
        print(f"✅ {name.capitalize()}: {simple_q}\n")
        seen.add(simple_q)
        dataset.append({"question": simple_q, "answer": base_answer})
        count += 1

        # Variantes avancées
        for name, sys_prompt, tpl in advanced_variants:
            print(f"🛠️ Reformulation ({name})…")
            user_prompt = tpl(simple_q)
            resp = generate_unique_question(sys_prompt, user_prompt, seen)
            if not resp:
                print(f"⚠️ Échec de la variante avancée '{name}'")
                continue
            print(f"✅ {name.capitalize()}: {resp}\n")
            seen.add(resp)
            dataset.append({"question": resp, "answer": base_answer})
            count += 1

    # Sauvegarde finale
    print("📚 Enregistrement du dataset JSONL…")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"🎉 Terminé : {len(dataset)} entrées générées dans '{output_path}'.")
