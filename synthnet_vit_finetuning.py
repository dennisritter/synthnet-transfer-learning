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
    AugMix,
)
# from PIL import Image
import click
import wandb
from modules.data import train_val_test_imagefolder
from modules.training import train_finetuning


@click.command()
@click.option(
    '--model',
    help="The model to use. Either a Huggingface model name or path to local model checkpoint directory.",
    type=str,
    required=True,
)
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
    '--train_layers',
    help="Which layers to train. Accepts a predefined choice of strings for implemented configurations.",
    type=click.Choice(['FULL', 'CLASS_HEAD']),
    default='full',
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
    _train_transforms = Compose([
        Resize(feature_extractor.size),
        CenterCrop(feature_extractor.size),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomApply([AugMix()], p=int(args.augmix)),
        RandAugment(),
        ToTensor(),
        Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ])
    _val_transforms = Compose([
        Resize(feature_extractor.size),
        CenterCrop(feature_extractor.size),
        ToTensor(),
        Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
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

    if args.train_layers == "CLASS_HEAD":
        for name, param in model.named_parameters():
            param.requires_grad = False
        model.classifier.weight.requires_grad = True
        model.classifier.bias.requires_grad = True

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


if __name__ == "__main__":
    main()
