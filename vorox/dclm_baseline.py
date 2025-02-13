
from datasets import load_dataset
from pprint import pprint
import torch

"""
Some notable datapoint fields: text, url
Example datapoint:
{
"bff_contained_ngram_count_before_dedupe": 20,
'fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob': 0.08555358648300171,
 'language_id_whole_page_fasttext': {'en': 0.9382619857788086},
 'metadata': {'Content-Length': '61769',
              'Content-Type': 'application/http; msgtype=response',
              'WARC-Block-Digest': 'sha1:O62W52EPMTRDM7TLTF2EUN6LW7D55B7S',
              'WARC-Concurrent-To': '<urn:uuid:55042cfb-1487-48d0-9c6a-04e8c186c250>',
              'WARC-Date': datetime.datetime(2014, 3, 13, 14, 49, 13),
              'WARC-IP-Address': '198.252.206.24',
              'WARC-Identified-Payload-Type': None,
              'WARC-Payload-Digest': 'sha1:VW3U6F6VXBN4HTCLVDUXUJ6IZWNOHWRV',
              'WARC-Record-ID': '<urn:uuid:a57efb1b-c977-4b38-8f2a-4aefb27e00ac>',
              'WARC-Target-URI': 'http://askubuntu.com/questions/114277/ubuntu-doesnt-boot-without-flash-drive',
              'WARC-Truncated': None,
              'WARC-Type': 'response',
              'WARC-Warcinfo-ID': '<urn:uuid:b5e3a9ec-69eb-4339-b321-68508bcb76fa>'},
 "previous_word_count": 268,
 "text": ...,
 'url': 'http://askubuntu.com/questions/114277/ubuntu-doesnt-boot-without-flash-drive',
 'warcinfo': 'robots: classic\r\n'
             'hostname: ip-10-183-142-35.ec2.internal\r\n'
             'software: Nutch 1.6 (CC)/CC WarcExport 1.0\r\n'
             'isPartOf: CC-MAIN-2014-10\r\n'
             'operator: CommonCrawl Admin\r\n'
             'description: Wide crawl of the web with URLs provided by Blekko '
             'for March 2014\r\n'
             'publisher: CommonCrawl\r\n'
             'format: WARC File Format 1.0\r\n'
             'conformsTo: '
             'http://bibnum.bnf.fr/WARC/WARC_ISO_28500_version1_latestdraft.pdf'}
}
"""

hf_dataset_name = "mlfoundations/dclm-baseline-1.0"


def dclm_baseline_gen():
    dataset = load_dataset(hf_dataset_name, split="train", streaming=True)
    for example in dataset:
        yield example


class BabyDCLM(torch.utils.data.Dataset):
    """
    Required capped number of batches for easy testing.
    """

    def __init__(self, macro_batch_size=1024, num_batches=100):
        self.dataset_gen = dclm_baseline_gen()
        self.macro_batch_size = macro_batch_size
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if idx >= self.num_batches:
            raise IndexError("Index out of bounds")
        return next(self.dataset_gen)


if __name__ == "__main__":
    dataset = load_dataset(hf_dataset_name, split="train", streaming=True)
    pprint(dataset.info)
    pprint(dataset.n_shards)
