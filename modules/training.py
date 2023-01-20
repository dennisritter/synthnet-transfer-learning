"""Defines training functions"""

import numpy as np
import torchvision.transforms as transforms
from transformers import TrainingArguments, Trainer
import evaluate
import wandb
from modules.data import collate_fn, UnNormalize


def train_finetuning(
    model,
    feature_extractor,
    train_ds,
    val_ds,
    test_ds,
    project_name: str,
    run_name: str,
    output_dir: str,
    seed: int,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    resume_id: str,
    resume: bool,
):
    if resume:
        wandb.init(project=project_name, id=resume_id, resume="must")
    else:
        wandb.init(project=project_name)
    if run_name:
        wandb.run.name = run_name

    unnormalize = UnNormalize(feature_extractor.image_mean, feature_extractor.image_std)
    wandb.log(
        {
            "train_examples": [
                wandb.Image(transforms.ToPILImage()(unnormalize(img)))
                for img in train_ds.shuffle(seed=seed)[:5]["pixel_values"]
            ]
        }
    )
    wandb.log(
        {
            "val_examples": [
                wandb.Image(transforms.ToPILImage()(unnormalize(img)))
                for img in val_ds.shuffle(seed=seed)[:5]["pixel_values"]
            ]
        }
    )
    wandb.log(
        {
            "test_examples": [
                wandb.Image(transforms.ToPILImage()(unnormalize(img)))
                for img in test_ds.shuffle(seed=seed)[:5]["pixel_values"]
            ]
        }
    )
    wandb.config.update(
        {
            "datasets": {
                "train_ds_samples": train_ds.num_rows,
                "val_ds_samples": val_ds.num_rows,
                "test_ds_samples": test_ds.num_rows,
            }
        }
    )
    training_args = TrainingArguments(
        # run_name=run_name,
        output_dir=f"{output_dir}/{wandb.run.name}",
        seed=seed,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        optim="adamw_torch",
        fp16=True,
        dataloader_num_workers=4,
        disable_tqdm=False,
        load_best_model_at_end=True,
        save_total_limit=3,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,
        report_to="wandb",
    )
    metrics = evaluate.combine([evaluate.load("accuracy")])

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metrics.compute(predictions=predictions, references=labels)

    # define trainer and start training loop
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )
    trainer.train(resume_from_checkpoint=resume)
    outputs = trainer.predict(test_ds)
    wandb.log(outputs.metrics)

    # log PR ROC_AUC CONFMAT -> Wandb lags
    # logits = outputs.predictions
    # predictions = np.argmax(logits, axis=-1)
    # ground_truth = np.array([sample['label'] for sample in test_ds])
    # wandb.log({"pr_curve": wandb.plot.pr_curve(
    #     ground_truth,
    #     logits,
    #     labels=[name for name in label2id.keys()],
    # )})
    # wandb.log({"roc_curve": wandb.plot.roc_curve(
    #     ground_truth,
    #     logits,
    #     labels=[name for name in label2id.keys()],
    # )})
    # wandb.log({
    #     "confusion_matrix": wandb.plot.confusion_matrix(
    #         probs=None,
    #         y_true=ground_truth,
    #         preds=predictions,
    #         class_names=[name for name in label2id.keys()],
    #     )
    # })

    print(f"{outputs.metrics=}")


# TODO: Refactor before using!
def train_hyperparam_search(
    config,
    model,
    train_ds,
    val_ds,
    test_ds,
    feature_extractor,
    output_dir: str,
):
    with wandb.init(config=config):
        # set sweep configuration
        config = wandb.config

        # set training arguments
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/{wandb.run.name}",
            # seed=,
            save_strategy="steps",
            evaluation_strategy="steps",
            logging_strategy="steps",
            eval_steps=25,
            logging_steps=25,
            save_steps=25,
            max_steps=config.steps_total_warmup[0],
            warmup_steps=config.steps_total_warmup[1],
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            optim="adamw_torch",
            fp16=True,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            remove_unused_columns=False,
            report_to="wandb",
        )
        metrics = evaluate.combine([evaluate.load("accuracy")])

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metrics.compute(predictions=predictions, references=labels)

        # define training loop
        trainer = Trainer(
            model,
            training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            tokenizer=feature_extractor,
        )

        # start training loop
        trainer.train()
        outputs = trainer.predict(test_ds)
        wandb.log(outputs.metrics)
        print(f"{outputs.metrics=}")
