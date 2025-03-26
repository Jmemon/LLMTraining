from datasets import load_dataset

DCLM_PREFIX = "dclm-baseline"
DCLM_BASELINE_REPO_ID = "mlfoundations/dclm-baseline-1.0"

def dclm_baseline():
    dataset = load_dataset(DCLM_BASELINE_REPO_ID, split='train', streaming=True)
    return dataset.map(lambda x: {"text": x["text"]})


if __name__ == "__main__":
    from vorox.utils import download_convert_and_upload_hf_to_s3, get_dclm_baseline_urls
    download_convert_and_upload_hf_to_s3("mlfoundations/dclm-baseline-1.0", "vorox-processed-train-data", 10, "dclm-baseline")
    print(get_dclm_baseline_urls("vorox-processed-train-data", "dclm-baseline"))
