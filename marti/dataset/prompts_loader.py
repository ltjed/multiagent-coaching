import random
import json
from tqdm import tqdm
from torch.utils.data import Dataset


def collate_with_none_labels(batch):
    """
    Custom collate function that handles None values in labels.

    DSBench tasks have answer=None because ground truth is stored separately in
    local filesystem (answer_dir). This collate function allows batching while
    preserving None labels.

    Args:
        batch: List of samples from PromptDatasetWithLabel

    Returns:
        Batched data with None labels preserved

    Example:
        >>> batch = [
        ...     {"prompt": "p1", "label": None, "indice": 0},
        ...     {"prompt": "p2", "label": None, "indice": 1}
        ... ]
        >>> result = collate_with_none_labels(batch)
        >>> result["label"]  # None (not a tensor)
    """
    from torch.utils.data._utils.collate import default_collate

    if not batch:
        return {}

    elem = batch[0]
    if isinstance(elem, dict):
        result = {}
        for key in elem:
            values = [d[key] for d in batch]

            # Handle None values - don't try to collate them into tensors
            if all(v is None for v in values):
                # All None → keep as None (DSBench case)
                result[key] = None
            elif any(v is None for v in values):
                # Mix of None and non-None → keep as list (shouldn't happen)
                result[key] = values
            else:
                # No None values → use default collation (convert to tensors)
                try:
                    result[key] = default_collate(values)
                except (TypeError, RuntimeError):
                    # Fallback: keep as list if default_collate fails
                    result[key] = values
        return result

    # Not a dict, use default collation
    return default_collate(batch)


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None, add_prompt_suffix=None, add_system_prompt=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if add_prompt_suffix is not None:
            chat += add_prompt_suffix
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        if add_system_prompt:
            chat = [{"role": "system", "content": add_system_prompt}] + chat
        prompt = apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if add_prompt_suffix is not None:
            prompt += add_prompt_suffix
        if input_template:
            prompt = input_template.format(prompt)
    return prompt


class PromptDatasetWithLabel(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
        add_prompt_suffix=None
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(
            self.strategy.args, "apply_chat_template", False)

        add_think_token = getattr(self.strategy.args, "add_think_token", 0)
        add_system_prompt = getattr(
            self.strategy.args, "add_system_prompt", None)
        if apply_chat_template and self.tokenizer is not None:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        indice = 0
        print(dataset)
        for data in tqdm(dataset, desc="Preprocessing data"):
            prompt = preprocess_data(data, input_template, input_key,
                                     apply_chat_template, add_prompt_suffix, add_system_prompt)
            if apply_chat_template and add_think_token != 0:
                prompt = prompt + "<think>"
            label = data[label_key]

            if isinstance(label, list):
                label = label[0]
            if isinstance(label, float) or isinstance(label, int):
                label = str(label)

            metadata_key = self.strategy.args.metadata_key
            sample = {"prompt": prompt, "label": label,
                      "indice": indice, "metadata": json.dumps(data.get(metadata_key, {}))}

            self.prompts.append(sample)
            indice += 1

        # Print sample prompts for logging (handle small datasets)
        sample_size = min(2, len(self.prompts))
        if sample_size > 0:
            for sample in random.sample(self.prompts, sample_size):
                print(sample)
                print("="*20)

    def get_all_prompts(self):
        return self.prompts

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
