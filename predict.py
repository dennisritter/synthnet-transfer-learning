"""finteune vit"""
from types import SimpleNamespace
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, Trainer, TrainingArguments
import evaluate
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)
from datasets import load_dataset
import click
from modules.data import collate_fn
import evaluate


@click.command()
@click.option(
    '--model',
    help="The model to use. Either a Huggingface model name or path to local model checkpoint directory.",
    type=str,
    required=True,
)
@click.option(
    '--test_ds',
    help="Test Dataset in Huggingface imagefolder structure.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
@click.option(
    '--batch_size',
    help='The batch size',
    type=int,
    show_default=True,
    default=16,
)
@click.option(
    '--workers',
    help='Number of workers for dataloader',
    type=int,
    show_default=True,
    default=4,
)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    test_ds = load_dataset("imagefolder", data_dir=args.test_ds, split="test")
    print(test_ds)

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)

    _val_transforms = Compose([
        Resize(feature_extractor.size),
        CenterCrop(feature_extractor.size),
        ToTensor(),
        Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ])

    def val_transforms(examples):
        print(examples)
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    test_ds.set_transform(val_transforms)

    id2label = {id: label for id, label in enumerate(test_ds.features['label'].names)}
    label2id = {label: id for id, label in id2label.items()}
    num_labels = len(id2label.keys())

    model = AutoModelForImageClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    metrics = evaluate.combine([evaluate.load("accuracy")])

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metrics.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir='out/test',
        per_device_eval_batch_size=args.batch_size,
        fp16=True,
        dataloader_num_workers=4,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model,
        training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )
    outputs = trainer.predict(test_ds)
    print(outputs)


if __name__ == "__main__":
    main()
