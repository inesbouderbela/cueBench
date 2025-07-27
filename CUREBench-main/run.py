def main():
    # ✅ Directly load from path
    path = "/content/cueBench/CUREBench-main/metadata_config_val.json"
    args = load_and_merge_config(path)

    # Extract values dynamically with fallback defaults
    output_file = getattr(args, 'output_file', "submission.csv")
    dataset_name = getattr(args, 'dataset')
    model_name = getattr(args, 'model_path', None) or getattr(args, 'model_name', None)
    

    print("\n" + "="*60)
    print("🏥 CURE-Bench Competition - Evaluation")
    print("="*60)

    # ✅ Initialize competition kit
    kit = CompetitionKit(config_path=path)

    print(f"Loading model: {model_name}")
    kit.load_model(model_name)

    # Show available datasets
    print("Available datasets:")
    kit.list_datasets()

    # Run evaluation
    print(f"Running evaluation on dataset: {dataset_name}")
    results = kit.evaluate(dataset_name)

    # Generate submission
    print("Generating submission with metadata...")
    submission_path = kit.save_submission_with_metadata(
        results=[results],
        filename=output_file,
        config_path=path,
        args=args
    )

    print(f"\n✅ Evaluation completed successfully!")
    print(f"📊 Accuracy: {results.accuracy:.2%} ({results.correct_predictions}/{results.total_examples})")
    print(f"📄 Submission saved to: {submission_path}")

    final_metadata = kit.get_metadata(path, args)
    print("\n📋 Final metadata:")
    for key, value in final_metadata.items():
        print(f"  {key}: {value}")
