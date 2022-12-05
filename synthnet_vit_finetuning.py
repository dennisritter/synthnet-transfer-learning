import os
import time
from types import SimpleNamespace
import numpy as np
from datasets import load_dataset
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
import evaluate
import torch
from torchvision.transforms import (CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomVerticalFlip, Resize, ToTensor)
# import torchvision.transforms as transforms
# from PIL import Image
import click
import wandb


@click.command()
@click.option(
    '--project_name',
    help="Project name this run belongs to. (WandB Project)",
    type=str,
    required=True,
)
@click.option(
    '--output_dir',
    help='Output directory path.',
    type=click.Path(),
    required=True,
)
@click.option(
    '--train_ds',
    help="Training Dataset in Huggingface imagefolder structure.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
@click.option(
    '--val_ds',
    help="Validation Dataset in Huggingface imagefolder structure. If no val_ds given, it will be split from the train_ds.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    '--test_ds',
    help="Test Dataset in Huggingface imagefolder structure.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
@click.option(
    '--run_name',
    help="Name for this training run.",
    type=str,
    required=False,
)
@click.option(
    '--mode',
    help="Either HP_Search or Finetuning mode.",
    type=click.Choice(['HP_SEARCH', 'FINETUNING']),
    required=False,
)
@click.option(
    '--checkpoint',
    help="Path to a pretrained checkpoint to load.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=False,
)
@click.option(
    '--split_val_size',
    help='Split size from train_ds used for validation (val_ds).',
    type=float,
    show_default=True,
    default=0.1,
)
@click.option(
    '--seed',
    help='random seed',
    type=int,
    show_default=True,
    default=42,
)
@click.option(
    '--num_train_epochs',
    help='TrainArguments.num_train_epochs',
    type=int,
    show_default=True,
    default=3,
)
@click.option(
    '--per_device_train_batch_size',
    help='TrainArguments.per_device_train_batch_size',
    type=int,
    show_default=True,
    default=16,
)
@click.option(
    '--per_device_eval_batch_size',
    help='TrainArguments.per_device_eval_batch_size',
    type=int,
    show_default=True,
    default=16,
)
@click.option(
    '--learning_rate',
    help='TrainArguments.learning_rate',
    type=float,
    show_default=True,
    default=2e-5,
)
@click.option(
    '--weight_decay',
    help='TrainArguments.weight_decay',
    type=float,
    show_default=True,
    default=1e-2,
)
@click.option(
    '--hps_train_size',
    help='Relative size of val_ds in HP_SEARCH mode [0,1]',
    type=float,
)
@click.option(
    '--hps_val_size',
    help='Relative size of val_ds in HP_SEARCH mode [0,1]',
    type=float,
)
@click.option(
    '--hps_run_count',
    help='Number of runs to start with different hyperparameters',
    type=float,
    show_default=True,
    default=10,
)
def main(**kwargs):
    # Parse click parameters and load config
    args = SimpleNamespace(**kwargs)
    wandb.login()
    ###############################################################
    ## LOADING DATA
    ###############################################################

    # load datasets from image directory (huggingface)
    # https://huggingface.co/docs/datasets/image_dataset
    # Ensure format ds_dir/label/file.png
    train_ds = load_dataset("imagefolder", data_dir=args.train_ds, split="train")
    if args.val_ds:
        # Use val_ds as defined by args
        val_ds = load_dataset("imagefolder", data_dir=args.val_ds, split="test")
    else:
        # split up training into training + validation
        splits = train_ds.train_test_split(test_size=args.split_val_size)
        train_ds = splits['train']
        val_ds = splits['test']
    test_ds = load_dataset("imagefolder", data_dir=args.test_ds, split="test")

    # map ids to labels and vice versa
    id2label = {id: label for id, label in enumerate(train_ds.features['label'].names)}
    label2id = {label: id for id, label in id2label.items()}

    ###############################################################
    ## PREPROCESSING DATA
    ###############################################################

    # load huggingface feature extractor to prepare data for the model
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    # Define Transforms
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    # NOTE: type(feature_extractor.size) changes from INT to DICT (transformers 4.24 -> 4.25)
    _train_transforms = Compose([
        Resize(feature_extractor.size),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
        normalize,
    ])

    _val_transforms = Compose([
        Resize(feature_extractor.size),
        CenterCrop(feature_extractor.size),
        ToTensor(),
        normalize,
    ])

    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    # Set the transforms
    train_ds.set_transform(train_transforms)
    val_ds.set_transform(val_transforms)
    test_ds.set_transform(val_transforms)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    # define training eval metric
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    ###############################################################
    ## DEFINE MODEL AND TRAINING CONFIGURATIONS
    ###############################################################

    num_labels = len(id2label.keys())
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_labels, id2label=id2label, label2id=label2id)

    def train_finetuning(
        run_name: str,
        output_dir: str,
        seed: int,
        num_train_epochs: int,
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        checkpoint: str,
    ):
        wandb.init(project=args.project_name)
        if run_name:
            wandb.run.name = run_name
        wandb.config.update({
            "datasets": {
                "train_ds": args.train_ds,
                "train_ds_samples": train_ds.num_rows,
                "val_ds": args.val_ds or args.train_ds,
                "val_ds_samples": val_ds.num_rows,
                "test_ds": args.test_ds,
                "test_ds_samples": test_ds.num_rows,
            }
        })
        # define train arguments
        # timestamp = time.strftime("%y%m%d%M%S", time.gmtime())
        training_args = TrainingArguments(
            # run_name=run_name,
            output_dir=f"{output_dir}/{wandb.run.name}",
            seed=seed,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_strategy='epoch',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optim="adamw_torch",
            fp16=True,
            # adam_beta1=0.9,
            # adam_beta2=0.999,
            # adam_epsilon=1e-8,
            # max_grad_norm=1.0,
            # dataloader_num_workers=0,
            disable_tqdm=False,
            load_best_model_at_end=True,
            save_total_limit=3,
            metric_for_best_model="accuracy",
            remove_unused_columns=False,
            report_to="wandb",
        )

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
        if args.checkpoint:
            trainer.train(resume_from_checkpoint=checkpoint)
        else:
            trainer.train()

        ###############################################################
        ## EVALUATION
        ###############################################################
        outputs = trainer.predict(test_ds)
        print(f"{outputs.metrics=}")

    def train_hyperparam_search(config=None):
        with wandb.init(config=config):
            # set sweep configuration
            config = wandb.config

            # set training arguments
            training_args = TrainingArguments(
                output_dir=f'{args.output_dir}/{wandb.run.name}',
                # seed=,
                save_strategy='epoch',
                evaluation_strategy='epoch',
                logging_strategy='epoch',
                num_train_epochs=config.epochs,
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
                report_to='wandb',
            )

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

    if args.mode == "FINETUNING":
        train_finetuning(
            run_name=args.run_name,
            output_dir=args.output_dir,
            seed=args.seed,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            checkpoint=args.checkpoint,
        )

    if args.mode == "HP_SEARCH":
        if args.hps_train_size:
            train_ds = train_ds.train_test_split(train_size=args.hps_train_size, stratify_by_column='label')['train']
        if args.hps_val_size:
            val_ds = val_ds.train_test_split(test_size=args.hps_val_size, stratify_by_column='label')['test']

        sweep_config = {
            'method': 'bayes',
            'metric': {
                'name': 'accuracy',
                'goal': 'maximize'
            },
            'parameters': {
                'epochs': {
                    'value': args.num_train_epochs
                },
                'batch_size': {
                    'values': [8, 16, 32, 64]
                },
                'learning_rate': {
                    'min': 1e-5,
                    'max': 1e-4
                },
                'weight_decay': {
                    'min': 0.01,
                    'max': 0.1
                },
            },
            'distribution': {}
        }
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.project_name)
        wandb.agent(sweep_id, train_hyperparam_search, count=20)


if __name__ == "__main__":
    main()