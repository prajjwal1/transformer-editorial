import random
import json
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np


from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from transformers import (
    EvalPrediction,
    Trainer,
    default_data_collator,
    TrainingArguments,
    HfArgumentParser
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

@dataclass
class DataArguments:
    fold: int = field()
    k: int = field()

parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

with open("edit_data.json", "r") as read_file:
    data = json.load(read_file)

df = pd.DataFrame(data)

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

def preprocess_function(examples):
    #  prompt = str(examples['body'])
    prompt = str(examples['abstract'])
    choice0, choice1, choice2 = str(examples['thesis']), str(examples['anti-thesis']), str(examples['third-option'])
    choices = [choice0, choice1, choice2]
    random.shuffle(choices)
    encoding = tokenizer([prompt, prompt, prompt], choices, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
    encoding["label"] = choices.index(choice0)
    return encoding


five_fold = {0: list(range(0, 169)),
             1: list(range(169, 338)),
             2: list(range(338, 507)),
             3: list(range(507, 676)),
             4: list(range(676, 845))}

ten_fold = {0: list(range(0, 84)),
            1: list(range(84, 168)),
            2: list(range(168, 252)),
            3: list(range(252, 336)),
            4: list(range(336, 420)),
            5: list(range(420, 504)),
            6: list(range(504, 588)),
            7: list(range(588, 672)),
            8: list(range(672, 756)),
            9: list(range(756, 845))}

if data_args.fold == 5:
    total_indexes = set(range(0, 845))
    valid_indexes = set(five_fold[data_args.k])
    train_indexes = total_indexes - valid_indexes
    train_indexes, valid_indexes = list(train_indexes), list(valid_indexes)
    assert len(train_indexes)+len(valid_indexes) <= 845
    assert len(train_indexes)+len(valid_indexes) > 840

if data_args.fold == 10:
    total_indexes = set(range(0, 845))
    valid_indexes = set(ten_fold[data_args.k])
    train_indexes = total_indexes - valid_indexes
    train_indexes, valid_indexes = list(train_indexes), list(valid_indexes)
    assert len(train_indexes)+len(valid_indexes) <= 845
    assert len(train_indexes)+len(valid_indexes) > 840


train_dataset = Dataset.from_pandas(df.iloc[train_indexes])
eval_dataset = Dataset.from_pandas(df.iloc[valid_indexes])
train_dataset = train_dataset.map(preprocess_function)
eval_dataset = eval_dataset.map(preprocess_function)

model = AutoModelForMultipleChoice.from_pretrained(model_args.model_name_or_path, return_dict=True)

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
)

if training_args.do_train:
    result = trainer.evaluate()
    print(result)
    trainer.train()
    trainer.save_model()
if training_args.do_eval:
    result = trainer.evaluate()
    print(result)
