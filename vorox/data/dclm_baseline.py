from datasets import load_dataset

DCLM_PREFIX = "dclm-baseline"
DCLM_BASELINE_REPO_ID = "mlfoundations/dclm-baseline-1.0"

def dclm_baseline():
    dataset = load_dataset(DCLM_BASELINE_REPO_ID, split='train', streaming=True)
    return dataset.map(lambda x: {"text": x["text"]})

