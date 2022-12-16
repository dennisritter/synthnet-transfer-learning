"""finteune vit"""
from types import SimpleNamespace
import random
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, set_seed, logging
import torch
from torchvision.transforms import (
    RandomApply,
    RandAugment,
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    ToTensor,
    Grayscale,
    AugMix,
)
# from PIL import Image
import click
import wandb
from modules.data import train_val_test_imagefolder
from modules.training import train_finetuning


@click.command()
@click.option(
    '--project_name',
    help="Project name this run belongs to. (WandB Project)",
    type=str,
    required=True,
)
@click.option(
    '--model',
    help="The model to use",
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
    '--seed',
    help='random seed',
    type=int,
    show_default=True,
    default=42,
)
@click.option(
    '--batch_size',
    help='The batch size',
    type=int,
    show_default=True,
    default=8,
)
@click.option(
    '--num_train_epochs',
    help='number of epochs to train',
    type=int,
    show_default=True,
    default=20,
)
@click.option(
    '--learning_rate',
    help='TrainArguments.learning_rate',
    type=float,
    show_default=True,
    default=3e-3,
)
@click.option(
    '--weight_decay',
    help='TrainArguments.weight_decay',
    type=float,
    show_default=True,
    default=3e-2,
)
@click.option(
    '--warmup_ratio',
    help='Warmup steps',
    type=float,
    show_default=True,
    default=0.1,
)
@click.option(
    '--workers',
    help='Number of workers for dataloader',
    type=int,
    show_default=True,
    default=4,
)
@click.option(
    '--grayscale',
    help='use grayscale transform for train and test loader',
    type=bool,
    show_default=True,
    default=False,
)
@click.option(
    '--augmix',
    help='use grayscale transform for train loader',
    type=bool,
    show_default=True,
    default=True,
)
def main(**kwargs):
    # Parse click parameters and load config
    args = SimpleNamespace(**kwargs)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)
    wandb.login()
    logging.set_verbosity_error()

    # load datasets
    train_ds, val_ds, test_ds = train_val_test_imagefolder(
        train_dir=args.train_ds,
        val_dir=args.val_ds,
        test_dir=args.test_ds,
    )

    # define feature_extractor for data preparation
    # TODO: add modelname to args
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)

    # Define Transforms
    # NOTE: type(feature_extractor.size) changes from INT to DICT (transformers 4.24 -> 4.25)
    _train_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomApply([AugMix()], p=int(args.augmix)),
            RandAugment(),
            RandomApply([Grayscale(3)], p=int(args.grayscale)),
            ToTensor(),
            Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
        ]
    )
    _val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            RandomApply([Grayscale(3)], p=int(args.grayscale)),
            ToTensor(),
            Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
        ]
    )

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

    ###############################################################
    ## DEFINE MODEL AND TRAINING CONFIGURATIONS
    ###############################################################
    id2label = {id: label for id, label in enumerate(train_ds.features['label'].names)}
    label2id = {label: id for id, label in id2label.items()}
    num_labels = len(id2label.keys())
    model = AutoModelForImageClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    if args.mode == "FINETUNING":
        train_finetuning(
            model=model,
            feature_extractor=feature_extractor,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            project_name=args.project_name,
            run_name=args.run_name,
            output_dir=args.output_dir,
            seed=args.seed,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            resume_id=args.resume_id,
            resume=args.resume,
        )

    # # TODO: fix sweep config
    # if args.mode == "HPS":
    #     sweep_config = {
    #         'method': 'grid',
    #         'metric': {
    #             'name': 'eval/accuracy',
    #             'goal': 'maximize'
    #         },
    #         'parameters': {
    #             'steps_total_warmup': {
    #                 'values': [(2000, 200)]
    #             },
    #             'batch_size': {
    #                 'value': args.batch_size
    #             },
    #             'learning_rate': {
    #                 'values': [1e-3, 3e-3, 0.01],
    #             },
    #             'weight_decay': {
    #                 'value': 0.03,
    #             }
    #         }
    #     }
    #     sweep_id = wandb.sweep(sweep=sweep_config, project=args.project_name)
    #     wandb.agent(sweep_id, train_hyperparam_search)


if __name__ == "__main__":
    main()
