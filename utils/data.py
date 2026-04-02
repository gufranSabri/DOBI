
import random
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset


class DataProcessor:
    def __init__(self, config, tokenizer, filepath, **kwargs):
        self.config    = config
        self.tokenizer = tokenizer
        self.filepath  = filepath

    def initializer(self):
        pass

    def line2data(self, indexed_example: tuple) -> list | None:
        raise NotImplementedError


class SmolTalkProcessor(DataProcessor):
    MAGPIE_SUBSET = "smol-magpie-ultra"
    MAGPIE_EXCLUDE_CATS = {
        "advice-seeking", "brainstorming", "creative-writing",
        "editing", "planning", "role-playing",
    }

    def line2data(self, indexed_example: tuple) -> list:
        _, ex = indexed_example
        messages = ex["messages"]

        # Category filter (hardcoded)
        category = ex.get("category")
        if category in self.MAGPIE_EXCLUDE_CATS:
            return []

        examples = []
        for idx in range(1, len(messages)):
            context  = messages[:idx]
            response = messages[idx]

            if response["role"] != "assistant":
                continue

            context_ids = self.tokenizer.apply_chat_template(
                context,
                tokenize=True,
                add_generation_prompt=True,
            )
            if not isinstance(context_ids, list):
                context_ids = context_ids["input_ids"]

            response_ids = self.tokenizer.encode(
                response["content"],
                add_special_tokens=False,
            )

            if len(context_ids) + len(response_ids) > self.config.MAX_LENGTH:
                break

            # Build input_ids and labels: mask context tokens with -100
            input_ids = context_ids + response_ids
            labels    = [-100] * len(context_ids) + response_ids

            examples.append({
                "input_ids": input_ids,
                "labels":    labels,
            })

        return examples


def build_datasets(args, tokenizer):
    args.logger("Loading smoltalk subsets …")

    use_all_train    = args.MAX_TRAIN_SAMPLES == -1
    n_val_per_subset = args.MAX_VAL_SAMPLES // len(args.DATASET_SUBSETS)

    if not use_all_train:
        total_needed       = args.MAX_TRAIN_SAMPLES + args.MAX_VAL_SAMPLES
        n_per_subset       = total_needed // len(args.DATASET_SUBSETS)
        n_train_per_subset = n_per_subset - n_val_per_subset

    train_shards, val_shards = [], []

    filepaths = [
        f"{args.DATASET_ID},{subset},train" for subset in args.DATASET_SUBSETS
    ] + [
        f"{args.DATASET_ID},{subset},test"  for subset in args.DATASET_SUBSETS
    ]

    subset_filepaths = [
        (f"{args.DATASET_ID},{subset},train", f"{args.DATASET_ID},{subset},test")
        for subset in args.DATASET_SUBSETS
    ]

    for subset, (train_fp, test_fp) in zip(args.DATASET_SUBSETS, subset_filepaths):
        for split_fp, shard_list, n_target, label in [
            (train_fp, train_shards, None if use_all_train else n_train_per_subset, "train"),
            (test_fp,  val_shards,   n_val_per_subset,                              "val"),
        ]:
            dataset_id, sub, split = split_fp.split(",")
            _dataset = load_dataset(dataset_id, sub, split=split).shuffle(seed=42)

            if label == "train" and not use_all_train:
                _dataset = _dataset.select(range(min(n_target, len(_dataset))))
            elif label == "val":
                _dataset = _dataset.select(range(min(n_val_per_subset, len(_dataset))))

            processor = SmolTalkProcessor(config=args, tokenizer=tokenizer, filepath=split_fp)
            processor.initializer()

            dataset = []
            for item in tqdm(
                enumerate(_dataset),
                desc=f"Loading {label} data from {split_fp}",
                total=len(_dataset),
            ):
                result = processor.line2data(item)
                if result is not None:
                    dataset.extend(result)

            under_budget = len(dataset) < (n_target or 0) and label == "train"
            args.logger(
                f"  {subset} [{label}]: {len(dataset)} examples"
                + (" ⚠ subset smaller than target" if under_budget else "")
            )

            shard_list.append(dataset)

    # Flatten, shuffle, cap
    def flatten_and_cap(shards, cap):
        flat = [ex for shard in shards for ex in shard]
        random.shuffle(flat)
        return flat[:cap] if cap != -1 else flat

    train_data = flatten_and_cap(train_shards, args.MAX_TRAIN_SAMPLES)
    val_data   = flatten_and_cap(val_shards,   args.MAX_VAL_SAMPLES)

    args.logger(f"  Total after length filter (≤{args.MAX_LENGTH} tokens): "
                f"{len(train_data)} train / {len(val_data)} val\n")

    # Wrap in HuggingFace Dataset for the collator
    train_tokenized = Dataset.from_list(train_data)
    val_tokenized   = Dataset.from_list(val_data)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=None,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    return train_tokenized, val_tokenized, data_collator