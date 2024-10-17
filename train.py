from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from dataloader import load_data
from eval import compute_metrics_currier
from hyperparam import Hyperparams as hp
from tokenizer import labelize_inputs

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(hp.pretrained_model_repo)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    input_dataset = load_data()

    labeled_inputs, labels_to_hanja_words, hanja_words_to_labels = labelize_inputs(input_dataset, tokenizer)
    compute_metrics = compute_metrics_currier(labels_to_hanja_words)

    model = AutoModelForTokenClassification.from_pretrained(
        hp.pretrained_model_repo, num_labels=len(labels_to_hanja_words), id2label=labels_to_hanja_words, label2id=hanja_words_to_labels
    )

    training_args = TrainingArguments(
        output_dir=hp.model_output_dir,
        learning_rate=hp.learning_rate,
        per_device_train_batch_size=hp.per_device_train_batch_size,
        per_device_eval_batch_size=hp.per_device_eval_batch_size,
        num_train_epochs=hp.num_train_epochs,
        weight_decay=hp.weight_decay,
        eval_strategy=hp.eval_strategy,
        save_strategy=hp.save_strategy,
        save_total_limit=hp.save_total_limit,
        load_best_model_at_end=hp.load_best_model_at_end,
        push_to_hub=hp.push_to_hub,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=labeled_inputs["train"],
        eval_dataset=labeled_inputs["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=hp.resume_from_checkpoint)

    if hp.push_to_hub:
        trainer.push_to_hub()
