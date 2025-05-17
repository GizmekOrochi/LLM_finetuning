def estimate_final_dataset_size(
    topics,
    num_questions_per_topic,
    basic_variants=1,
    advanced_variants=3
):
    num_topics = len(topics)
    pairs_per_topic = num_questions_per_topic * (num_questions_per_topic - 1) // 2
    variants_per_pair = basic_variants + advanced_variants
    return num_topics * pairs_per_topic * variants_per_pair

if __name__ == "__main__":
    topics = [
        "Préhistoire",
        "Antiquité égyptienne",
        "Antiquité gréco-romaine",
        "Empire romain",
        "Empire byzantin",
        "Haut Moyen Âge",
        "Bas Moyen Âge",
        "Croisades",
        "Renaissance",
        "Réforme protestante",
        "Guerre de Cent Ans",
        "Monarchie absolue",
        "Révolution française",
        "Époque napoléonienne",
        "Révolution industrielle",
        "Première Guerre mondiale",
        "Seconde Guerre mondiale",
        "Guerre froide",
        "Décolonisation",
        "Mondialisation contemporaine"
    ]
    num_q = 50

    total = estimate_final_dataset_size(
        topics=topics,
        num_questions_per_topic=num_q,
        basic_variants=1,
        advanced_variants=5
    )
    print(f"Expected total examples (lines) in final dataset: {total}")
