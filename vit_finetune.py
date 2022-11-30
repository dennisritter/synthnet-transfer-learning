import time
import numpy as np
from datasets import load_dataset
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
import evaluate
import torch
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomVerticalFlip,
                                    Resize, 
                                    ToTensor)
# import torchvision.transforms as transforms
# from PIL import Image
import click

@click.command()
@click.option(
    '--run_name',
    help="Name for this training run. Timestamp will be appended on runtime.",
    type=str,
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
    help="Validation Dataset in Huggingface imagefolder structure.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
@click.option(
    '--test_ds',
    help="Test Dataset in Huggingface imagefolder structure.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
@click.option(
    '--checkpoint',
    help="Path to a pretrained checkpoint to load.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=False,
)
@click.option(
    '--out_root_dir',
    help='Output root directory path. Timestamp will be appended on runtime.',
    type=click.Path(),
    required=True,
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
    default=10,
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



def main(**kwargs):
    # Parse click parameters and load config
    config = load_config(**kwargs)
    
    ###############################################################
    ## LOADING DATA
    ###############################################################

    # load a local dataset from image directory (huggingface)
    # https://huggingface.co/docs/datasets/image_dataset
    DATASET_TRAIN_NAME = '7054-12-300-l_drucker_se_su_st_st_512_32'
    DATASET_TRAIN_DIR = f'data/{DATASET_TRAIN_NAME}'

    train_ds = load_dataset("imagefolder", data_dir=DATASET_TRAIN_DIR, split="train")
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']

    DATASET_TEST_NAME = 'topex-real-123_pb_256'
    DATASET_TEST_DIR = f'data/{DATASET_TEST_NAME}'

    test_ds = load_dataset("imagefolder", data_dir=DATASET_TEST_DIR, split="test")
    # We use train_test_split function to get val and test sets
    # splits = test_ds.train_test_split(test_size=0.9, stratify_by_column='label')
    # val_ds = splits['train']
    # test_ds = splits['test']

    id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
    label2id = {label:id for id,label in id2label.items()}

    ###############################################################
    ## PREPROCESSING DATA
    ###############################################################

    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    # Define Transforms
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    _train_transforms = Compose(
            [   
                Resize(feature_extractor.size),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    _val_transforms = Compose(
            [
                Resize(feature_extractor.size),
                CenterCrop(feature_extractor.size),
                ToTensor(),
                normalize,
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


    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    ###############################################################
    ## DEFINE MODEL
    ###############################################################

    num_labels = len(id2label.keys())
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                    num_labels=num_labels,
                                                    id2label=id2label,
                                                    label2id=label2id)

    # define train arguments
    timestamp = time.strftime("%y%m%d%M%S",time.gmtime())
    args = TrainingArguments(
        run_name=f"test/synthnet/vit/{DATASET_TRAIN_NAME}-{timestamp}",
        output_dir=f"out/vit/{DATASET_TRAIN_NAME}-{timestamp}",
        seed=42,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=1e-2,
        optim="adamw_torch",
        # adam_beta1=0.9,
        # adam_beta2=0.999,
        # adam_epsilon=1e-8,
        # max_grad_norm=1.0,
        # dataloader_num_workers=0,
        disable_tqdm=False,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='logs',
        remove_unused_columns=False,
        report_to="wandb"
        # report_to=None
    )

    # define training eval metric
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # define trainer and start training loop
    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )
    trainer.train()

    ###############################################################
    ## EVALUATION
    ###############################################################

    outputs = trainer.predict(test_ds)
    print(f"{outputs.metrics=}")

if __name__ == "__main__":
    main()