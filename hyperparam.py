class Hyperparams:
    # Data
    maxlen = 500    # The max string length (not token numbers) of input
    labelizer_batch_size = 50
    test_ratio = 0.01   # The ratio of test data in total data

    # Model
    pretrained_model_repo = "klue/roberta-base"
    model_output_dir = "roberta-base-hangul-2-hanja"
    user_repo = "opengl106"

    # Learning
    learning_rate = 2e-5
    num_train_epochs = 60
    weight_decay = 0.01
    per_device_train_batch_size = 32    # 64 for small model and 32 for middle
    per_device_eval_batch_size = 32

    # Saving
    eval_strategy = "epoch"
    save_strategy = "epoch"
    save_total_limit = 5
    load_best_model_at_end = False
    push_to_hub = True
    resume_from_checkpoint = True
