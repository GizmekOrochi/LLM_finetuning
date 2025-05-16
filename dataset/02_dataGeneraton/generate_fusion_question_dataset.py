import json
from itertools import combinations
from ollama import Client

# Activer l'utilisation du GPU si disponible
USE_GPU = True
MODEL_NAME = "llama3.2:latest"

def get_ollama_options():
    opts = {
        "temperature": 0.2,
        "top_p": 0.9,
        "num_predict": 100,
    }
    if USE_GPU:
        # selon la version du client Ollama, utilisez "device" ou "gpu"
        opts["device"] = "cuda"
        # opts["gpu"] = True
    return opts

client = Client()

questions_fr = [
    "Quand a eu lieu la Révolution française ?",
    "Qui était Napoléon Bonaparte ?",
    "En quelle année a eu lieu la prise de la Bastille ?",
    "Qui était Marie-Antoinette ?",
    "Quel traité a mis fin à la Première Guerre mondiale ?"
]

# Génère un texte unique non présent dans 'seen', ou None si échec après max_attempts
def generate_unique_question(system_prompt: str,
                             user_prompt: str,
                             seen: set,
                             max_attempts: int = 10) -> str | None:
    for attempt in range(1, max_attempts + 1):
        resp = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            options=get_ollama_options()
        )
        text = resp.message.content.strip()
        if text not in seen:
            return text
        print(f"⚠️ Tentative {attempt}/{max_attempts}: doublon, retry…")
    print(f"❌ Échec après {max_attempts} tentatives pour le prompt : {user_prompt}")
    return None

# Définitions des variantes
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
    ("ouverte",    sys_rephrase, lambda q: f"Transforme cette question en question ouverte pour lancer la discussion : « {q} »"),
    ("academique", sys_rephrase, lambda q: f"Rédige cette question dans un style académique et formel : « {q} »"),
    ("fermee",     sys_rephrase, lambda q: f"Reformule en question fermée (oui/non) : « {q} »"),
    ("niveau_ce2", sys_rephrase, lambda q: f"Adapte cette question pour un élève de CE2 : « {q} »"),
    ("suivi",      sys_rephrase, lambda q: f"À partir de « {q} », formule une question de suivi pour approfondir le sujet"),
    ("passive",    sys_rephrase, lambda q: f"Reformule cette question à la voix passive : « {q} »"),
    ("comparative",sys_rephrase, lambda q: f"Transforme cette question pour comparer deux entités : « {q} »")
]

pairs = list(combinations(questions_fr, 2))
dataset = []

count = 0
for idx, (q1, q2) in enumerate(pairs, start=1):
    print("\n" + "="*60)
    print(f" 🤖 Exemple {idx}/{len(pairs)}  Réponses générées: {count}")
    print("="*60)
    print(f"Question 1: {q1}\nQuestion 2: {q2}\n")

    seen = set()
    results = {}
    base_answer = f"{q1} {q2}"

    # Variantes de base
    for name, sys_prompt, tpl in basic_variants:
        print(f"🛠️ Génération ({name})…")
        user_prompt = tpl(q1, q2)
        resp = generate_unique_question(sys_prompt, user_prompt, seen)
        if resp is None:
            print(f"⚠️ Échec de la variante de base '{name}', on passe à l'exemple suivant.\n")
            break
        count += 1
        print(f"✅ {name.capitalize()}: {resp}\n")
        results[name] = resp
        seen.add(resp)
        dataset.append({"question": resp, "answer": base_answer})
    else:
        # Exécuté uniquement si toutes les variantes de base réussissent
        complex_q = results['simple']
        # Variantes avancées
        for name, sys_prompt, tpl in advanced_variants:
            print(f"🛠️ Reformulation ({name})…")
            user_prompt = tpl(complex_q)
            resp = generate_unique_question(sys_prompt, user_prompt, seen)
            if resp is None:
                print(f"⚠️ Échec de la variante avancée '{name}', on continue les autres.\n")
                continue
            count += 1
            print(f"✅ {name.capitalize()}: {resp}\n")
            results[name] = resp
            seen.add(resp)
            dataset.append({"question": resp, "answer": base_answer})

# Enregistrement final
print("📚 Enregistrement du dataset JSONL…")
with open("fusion_fr_dataset.jsonl", "w", encoding="utf-8") as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print(f"🎉 Terminé : {len(dataset)} entrées générées dans 'fusion_fr_dataset.jsonl'.")