import torch

from transformers import TrainingArguments, HfArgumentParser, BertTokenizerFast, BertModel
from dataclasses import dataclass
from typing import AnyStr, Dict, Union, Any, Optional, List, Tuple
from datasets import DatasetDict
from datasets import load_metric
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import nested_detach
from torch import nn

from src.grounded_datasets import get_dataset
from src.models import get_model


accuracy_metric = load_metric("accuracy")


def compute_metrics(eval_prediction):
    pred = eval_prediction.predictions.argmax(-1)
    acc = accuracy_metric.compute(predictions=pred, references=eval_prediction.label_ids)

    return acc


class CustomTrainer(Trainer):

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Custom prediction step because labels are computed dynamically by our model, rather than specified by the
        dataset.  Remaining docstring below copied from original Trainer method.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
            else:
                logits = outputs[1:]

            logits = nested_detach(logits)
            if len(logits) == 1:
                logits = logits[0]

            # logits has shape B x num cands, of which the 0-th candidate is the ground truth
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

            return (loss, logits, labels)


def get_tokenizer(mname: AnyStr):
    tokenizer = BertTokenizerFast.from_pretrained(mname)

    return tokenizer


@dataclass
class CustomArguments:
    dataset = "pc.original"
    mname: Optional[str] = "memnet"  # Which model architecture you want to use
    dataset_disk_path: Optional[str] = None  # Overrides get_dataset method if set - must be DatasetDict
    pretrained_model: str = "huawei-noah/TinyBERT_General_4L_312D"  # Path to pretrained BERT model
    evaluate_responses: bool = True  # Whether to eval responses or knowledge - recommend True for WoW and False for PC


parser = HfArgumentParser([CustomArguments, TrainingArguments])
custom_args, training_args = parser.parse_args_into_dataclasses()


def main():
    tokenizer = get_tokenizer(mname=custom_args.pretrained_model)

    if custom_args.dataset_disk_path is None:
        dataset_dict = get_dataset(dataset_name=custom_args.dataset, tokenizer=tokenizer)
    else:
        dataset_dict = DatasetDict.load_from_disk(custom_args.dataset_disk_path)

    model = get_model(model=custom_args.mname, mname=custom_args.pretrained_model)
    model.bert2 = BertModel.from_pretrained(custom_args.pretrained_model)
    model.evaluate_responses = custom_args.evaluate_responses

    # Initialise weights for polyencoder codes following:
    # https://github.com/facebookresearch/ParlAI/blob/master/parlai/agents/transformer/polyencoder.py#L407
    try:
        torch.nn.init.normal_(model.codes.weight, 312 ** -0.5)
    # Throws error if using non-polyencoder model, in which case we do nothing.
    except:
        pass

    model.tokenizer = tokenizer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["valid"],
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
