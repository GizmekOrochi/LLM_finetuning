from generateNewSimpleQuestions import run_generate_questions
from batch_fusion_generator import run_fusion_pipeline


def build_all_datasets(
    topics,
    num_questions_per_topic,
    question_output_dir="dataset/question_groups",
    fusion_temp_dir="dataset/fusion_datasets",
    final_output_path="dataset/batch_fusion_generator/batch_fusion_dataset.jsonl",
    model_name="llama3.2:latest",
    use_gpu=True
):
    """
    1. Pour chaque thème dans `topics`, génère un jeu de questions simples.
    2. Exécute ensuite le pipeline de fusion pour produire le dataset final.
    """
    # Génération des groupes de questions par thème
    for topic in topics:
        print(f"\n📚 Génération des questions pour le thème « {topic} »...")
        run_generate_questions(
            topic=topic,
            num_questions=num_questions_per_topic,
            output_dir=question_output_dir,
            model_name=model_name,
            use_gpu=use_gpu
        )

    # Fusion finale
    print("\n🔗 Fusion des datasets générés en un seul fichier...")
    run_fusion_pipeline(
        question_folder=question_output_dir,
        temp_folder=fusion_temp_dir,
        merged_output_path=final_output_path
    )
    print("\n✅ Pipeline complet terminé.")


if __name__ == '__main__':
    # Exemple de configuration
    topics = [
        "histoire de France",
        "géographie mondiale",
        "sciences naturelles",
        "technologies modernes"
    ]
    num_questions = 2

    build_all_datasets(
        topics=topics,
        num_questions_per_topic=num_questions,
        question_output_dir="dataset/question_groups",
        fusion_temp_dir="dataset/fusion_datasets",
        final_output_path="dataset/batch_fusion_generator/batch_fusion_dataset.jsonl",
        model_name="llama3.2:latest",
        use_gpu=False  # passer à True si GPU disponible
    )
