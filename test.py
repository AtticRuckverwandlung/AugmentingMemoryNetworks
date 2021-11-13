"""
Once the models have been trained, we can evaluate them here.
"""
from src.models import get_model
from datasets import Dataset, load_metric
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
import torch
import numpy as np
from typing import Optional, AnyStr
from dataclasses import dataclass



@dataclass
class CustomArguments:
    test_dataset_path: AnyStr  # Path to test dataset
    pretrained_model: AnyStr  # Path to model you have trained
    mname: Optional[AnyStr] = "memnet"  # Which model architecture you want to use
    evaluate_responses: bool = True  # Whether to eval responses or knowledge - recommend True for WoW and False for PC
    batch_size: int = 2
    device: AnyStr = "cuda"


parser = HfArgumentParser([CustomArguments])
args = parser.parse_args([])

accuracy_metric = load_metric("accuracy")


def compute_metrics(eval_prediction):
    pred = eval_prediction.predictions.argmax(-1)
    acc = accuracy_metric.compute(predictions=pred, references=eval_prediction.label_ids)

    return acc


def main():
    model = get_model(args.mname, args.pretrained_model).to(args.device)
    model.evaluate_responses = args.evaluate_responses
    model.eval()

    cols = [
            "x_input_ids",
            "x_attention_mask",
            "y_input_ids",
            "y_attention_mask",
            "z_input_ids",
            "z_attention_mask",
            "lengths",
            "candidate_input_ids",
            "candidate_attention_mask"
        ]

    # Add more metrics here if desired
    metrics = {"accuracy": []}

    dataset = Dataset.load_from_disk(args.test_dataset_path)
    dataset.set_format(type="pt", columns=cols)
    num_samples = len(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            bsz = batch["x_input_ids"].shape[0]

            # 3D dims have format
            for key in ["z_input_ids", "z_attention_mask", "candidate_input_ids", "candidate_attention_mask"]:
                batch[key] = torch.stack(batch[key]).transpose(0, 1)

            # Send everything to device
            for key in cols:
                batch[key] = batch[key].to(args.device)

            outputs = model(**batch)
            labels = torch.zeros(bsz, dtype=torch.long).to(args.device)

            pred = outputs.logits.argmax(-1)
            acc = accuracy_metric.compute(predictions=pred, references=labels)
            metrics["accuracy"].append(acc["accuracy"])

            if i % 100 == 0:
                for metric, results in metrics.items():
                    print(f"{metric}: {np.mean(results)}; Samples completed: {(i+1)*args.batch_size}/{num_samples}.", )

        for metric, results in metrics.items():
            print(f"FINAL:  {metric}: {np.mean(results)}.")
