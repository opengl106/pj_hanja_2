class Hyperparams:
    maxlen = 500    # The max string length (not token numbers) of input
    labelizer_batch_size = 50
    test_ratio = 0.01   # The ratio of test data in total data

    learning_rate = 2e-5
    num_train_epochs = 20
    weight_decay = 0.01
    per_device_train_batch_size = 64    # 64 for small model and 32 for middle
    per_device_eval_batch_size = 64

    model_repo = "klue/roberta-small"
    output_dir = "roberta-small-hangul-2-hanja"

    eval_strategy = "epoch"
    save_strategy = "epoch"
    load_best_model_at_end = True
    push_to_hub = False
