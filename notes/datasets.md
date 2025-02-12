
One big learning is that the research community is very interested in is how best to curate effective LLM training datasets. 
Which can also be thought of as how best to filter text data for LLM training.


## DCLM-baseline
[S3](http://data.commoncrawl.org/contrib/datacomp/DCLM-baseline/index.html) | [HuggingFace](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | [GitHub](https://github.com/mlfoundations/dclm?tab=readme-ov-file) | [ArXiv](https://arxiv.org/pdf/2406.11794)

The point of the DCLM project is to facilitate building a better understanding of building high-quality datasets. The main dataset is DCLM-pool, which is the CommonCrawl (with boilerplate HTML extracted better using resiliparse). The DCLM-baseline is the project's proof of concept project that DCLM-pool can be filtered into a high-quality dataset.

### What's in it?
Contains 2-3T tokens. It is a filtered de-duplicated subset of CommonCrawl with model-based quality scoring.

Process:
1. HTML text extraction (resiliparse)
2. Heuristic filtering (language ID, length checks, repetition checks, domain blacklists)
3. Global deduplication (bloom filter)
4. Model-based filtering with fasttext – train binary classifier with equal split pos/neg examples from RefinedWeb.

## TheStack v2 – Dedup
[HuggingFace](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup) | [ArXiv](https://arxiv.org/pdf/2402.19173)

Code data.

### What's in it?
6.5Gb of data. 784M files.

Source Code
- Software Heritage archive – filtered, classified by language – extract most recently crawled version of github repos, retaining only main branch, and constructing directory structure

Notebooks
- Software Heritage archive – transformed to scripts
- Kaggle – Notebooks (transformed to scripts)

GitHub Issues + Pull Requests
- GitHub archive

Documentation
- Package Managers (npm, pypi, cargo, conda, rubygems, go packages, etc.) – use libraries.io to get most popular libraries across platforms, extract docs from READMEs, Read the Docs (best) if available, or whatever else
- Websites – MDN web docs, tensorflow docs, linux docs, LLVM doc, huggingface docs, etc.
- Free Programming Books – [Link](https://github.com/EbookFoundation/free-programming-books)
- RefinedWeb, OSCAR, esCorpius – datasets that contain some code-relevant info. Extracted code-relevant data with regex.

Intermediate Representations
- LLVM on github
- generated – compile ≈ 4M sources in (size and performance optimized mode -OZ -O3) – C, C++, Objective-C (clang) – Python (codon) – Rust (rustc) – Haskell (ghc) – Swift (swiftc) – Go (gollvm) – D (ldc) – Fortran (flang) – Nim (nlvm)

Small High-Quality Datasets
- APPS (train) – 5000 examples – text2code benchmark
- Code Contest – 13_000 examples – text2code benchmark
- GSM8K (train) – 7_000 examples – math reasoning benchmark
- GSM8K (SciRel) – 110_000 examples – augmented version of GSM8K (train)
- Deepmind mathematics – 110M short examples – synthetic dataset of math QAs across domains
- Rosetta Code – 1100 examples – everyday programming tasks with solutions in as many languages as possible
- MultiPL-T – 200_000 examples – high-quality Rust/Lua/OCaml from translating extracted python and validating with unit tests
- Proofsteps – ? + 253K examples – part of AlgebraicStack, proofsteps-lean (3k), proofsteps-isabelle (250K)

Natural Language
- StackExchange Archive – 10B tokens
- ArXiv subset of RedPajama – 30B tokens
- Wikipedia subset of RedPajama - 6B tokens
- OpenWebMath – derived from CommonCrawl – 15B tokens

## Dolma
[HuggingFace](https://huggingface.co/datasets/allenai/dolma) | [GitHub](https://github.com/allenai/dolma) | [ArXiv](https://arxiv.org/pdf/2402.00159) | [Blog](https://allenai.org/blog/olmo-1-7-7b-a-24-point-improvement-on-mmlu-92b43f7d269d)

AllenAI's dataset for training OLMo. Contains 3T tokens. Designed to be a high-quality totally open dataset for training LLMs. The Github contains code to re-create the dataset as well as a toolkit for high-performance language data processing.

### What's in it?
| Source | Tokens |
| --- | :---: |
| CommonCrawl | 2.5T |
| GitHub | 400B |
| Reddit | 100B |
| Semantic Scholar | 100B |
| Project Gutenberg | 6B |
| Wikipedia, Wikibooks | 4B |

## RedPajama
[HuggingFace](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) | [GitHub](https://github.com/togethercomputer/RedPajama-Data) | [ArXiv](https://arxiv.org/pdf/2411.12372)

RedPajama-v1 is an open reproduction of Llama's training dataset. RedPajama-v2 is a massive web-only with raw unfiltered text data as well as quality-signals and metadata.

### What's in it?

#### RedPajama-V1

Recreates seven main sources from the first LLaMA release:
- CommonCrawl (≈878B tokens)
- C4 (≈175B)
- GitHub (≈59B)
- Books (≈26B)
- arXiv (≈28B)
- Wikipedia (≈24B)
- StackExchange (≈20B)

Total tokens: ~1.2T

Process:
- CommonCrawl snapshots filtered with a fastText classifier trained on Wikipedia references
- GitHub files deduplicated and restricted to known OSI licenses (Apache, BSD, MIT), removing obviously low-quality code with heuristics
- Book sources come from PG19 (Project Gutenberg) plus Books3 (the latter was later removed due to copyright concerns)
- arXiv (LaTeX sources, stripped of macros/bibliography)
- Wikipedia (multilingual, 20 languages)
- StackExchange (28 largest sub-sites, minimal cleaning)

#### RedPajama-V2
- Strictly web-only content from CommonCrawl (2014–2023).
- ~84 CC snapshots → ~100T tokens across 113B raw documents.
- Covers five languages (English, French, German, Spanish, Italian).
- Minimal cleaning (just boilerplate removal via CCNet).
- Provides 46+ quality signals per document (fractions of repeated n-grams, natural-language “look,” blocklist matches, ML-based “importance,” dedup signatures, etc.).
- Lets you define your own data cleaning strategy (e.g. C4-style, Gopher-style, etc.).
- Fuzzy and exact dedup metadata available.
- Key Goal: Unlike monolithic pre-curated web corpora (C4, RefinedWeb, etc.), V2 is deliberately unfiltered + richly annotated, so that researchers can choose how strict or lenient to be in their data filtration.
