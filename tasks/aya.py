"""
Aya Dataset by Cohere For AI — multilingual instruction following.
https://huggingface.co/datasets/CohereForAI/aya_dataset

~204K human-curated multilingual instruction-response pairs across 65 languages.
All languages are loaded — the dataset is curated quality throughout.
Note: Aya's "language" field uses full names ("Arabic", "Chinese Simplified") not ISO
codes, so filtering by ISO code would silently exclude everything. Load all.
"""

from datasets import load_dataset
from tasks.common import Task


class Aya(Task):
    """
    Aya multilingual instruction-following dataset.
    Each example is a single-turn conversation: user instruction → assistant response.
    """

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ("train", "test"), "Aya split must be 'train' or 'test'"
        hf_split = "train" if split == "train" else "test"
        ds = load_dataset("CohereForAI/aya_dataset", split=hf_split)
        self.examples = list(ds)
        import random
        random.Random(42).shuffle(self.examples)

    def num_examples(self):
        return len(self.examples)

    def get_example(self, index):
        row = self.examples[index]
        instruction = row.get("inputs", row.get("instruction", "")).strip()
        response = row.get("targets", row.get("output", "")).strip()
        assert instruction, f"Empty instruction in Aya row {index}"
        assert response, f"Empty response in Aya row {index}"
        return {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
        }
