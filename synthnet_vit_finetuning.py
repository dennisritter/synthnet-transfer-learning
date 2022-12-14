from types import SimpleNamespace
import numpy as np
from datasets import load_dataset
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
import evaluate
import torch
from torchvision.transforms import (RandAugment, CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomVerticalFlip, Resize, ToTensor)
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
    '--resume',
    help="Do you want to resume a former run?",
    type=bool,
    required=False,
    show_default=True,
    default=False,
)
@click.option(
    '--resume_id',
    help="WandB run id you want to resume",
    type=str,
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
    '--total_steps',
    help='Total training steps',
    type=int,
    show_default=True,
    default=2500,
)
@click.option(
    '--warmup_steps',
    help='Warmup steps',
    type=int,
    show_default=True,
    default=200,
)
@click.option(
    '--batch_size',
    help='The batch size',
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
def main(**kwargs):
    # Parse click parameters and load config
    args = SimpleNamespace(**kwargs)
    wandb.login()
    ###############################################################
    ## LOADING DATA
    ###############################################################

    # load datasets from image directory (huggingface)
    # https://huggingface.co/docs/datasets/image_dataset
    # - Ensure format ds_dir/label/filename_SPLIT.png
    # - Each filename has to include the split name (e.g.: myname_test, train_my_name, my_val_name)
    train_ds = load_dataset("imagefolder", data_dir=args.train_ds, split="train")
    # Either use given val dataset or else split up training into training + validation
    if args.val_ds:
        val_ds = load_dataset("imagefolder", data_dir=args.val_ds, split="validation")
    else:
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
        CenterCrop(feature_extractor.size),
        RandAugment(),
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
        resume_id: str,
        resume: bool,
    ):
        if resume:
            wandb.init(project=args.project_name, id=resume_id, resume="must")
        else:
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
        trainer.train(resume_from_checkpoint=args.resume)
        outputs = trainer.predict(test_ds)
        wandb.log(outputs.metrics)
        print(f"{outputs.metrics=}")

    def train_hyperparam_search(config=None):
        with wandb.init(config=config):
            # set sweep configuration
            config = wandb.config

            # set training arguments
            training_args = TrainingArguments(
                output_dir=f'{args.output_dir}/{wandb.run.name}',
                # seed=,
                save_strategy='steps',
                evaluation_strategy='steps',
                logging_strategy='steps',
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
            outputs = trainer.predict(test_ds)
            wandb.log(outputs.metrics)
            print(f"{outputs.metrics=}")

    if args.mode == "FINETUNING":
        train_finetuning(
            run_name=args.run_name,
            output_dir=args.output_dir,
            seed=args.seed,
            total_steps=args.total_steps,
            warmup_steps=args.warmup_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            resume_id=args.resume_id,
            resume=args.resume,
        )

    if args.mode == "HP_SEARCH":
        sweep_config = {
            'method': 'grid',
            'metric': {
                'name': 'eval/accuracy',
                'goal': 'maximize'
            },
            'parameters': {
                'steps_total_warmup': {
                    'values': [(2000, 200)]
                },
                'batch_size': {
                    'value': args.batch_size
                },
                'learning_rate': {
                    'values': [1e-3, 3e-3, 0.01],
                },
                'weight_decay': {
                    'value': 0.03,
                },
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.project_name)
        wandb.agent(sweep_id, train_hyperparam_search)


if __name__ == "__main__":
    main()