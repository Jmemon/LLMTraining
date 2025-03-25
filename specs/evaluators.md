
# Evaluators

## High-Level Objective
Implement evaluators for the following datasets:
- MMLU
- GSM8K
- GSM-Symbolic
That are easy to instantiate and run on a model.

## Mid-Level Objectives
- Create EvaluatorBase class and file.
- Create MMLUEvaluator class and file.
- Create GSM8KEvaluator class and file.
- Create GSMSymbolicEvaluator class and file.
- Create EvaluatorBuilder class with build method and file.

## Implementation Notes
Use the HuggingFace `datasets` library to load and process the datasets.
Stream all datasets, do not download them locally. 
Create a few-shot prompt template for each evaluator based on the structure of the dataset rows. 
Applied to every row using dataset.map.
LEAVE NOTHING UNIMPLEMENTED.

## Context
### Beginning Context
vorox/configs.py (read-only)

### Ending Context
vorox/configs.py (read-only)
vorox/evaluators/evaluator_base.py
vorox/evaluators/mmlu_evaluator.py
vorox/evaluators/gsm8k_evaluator.py
vorox/evaluators/gsmsymbolic_evaluator.py
vorox/evaluators/builder.py

## Low-Level Tasks
1. Create EvaluatorBase class and file.
```aider
CREATE vorox/evaluators/evaluator_base.py:
    CREATE class EvaluatorBase:
        repo_id: str
        name: EvaluatorType
        performance_breakdown: Dict[str, Dict[str, int]]  # {subset_name: {num_correct: int, num_total: int}}
        dataloader: torch.utils.data.DataLoader

        CREATE def __init__(self, config: EvaluatorsConfig):
            """
            Initialize the evaluator.
            """
            pass

        CREATE def __call__(self, model: nn.Module) -> Dict[str, Dict[str, int]]:
            """
            Iterate over the dataloader, for each row putting together a prompt and getting a response from the model.
            Evaluate the response against the ground truth.
            Update the performance_breakdown dictionary.
            Return the performance_breakdown dictionary.
            """
            pass
```
1. Create MMLUEvaluator class and file.
```aider
CREATE vorox/evaluators/mmlu_evaluator.py:
    CREATE class MMLUEvaluator(EvaluatorBase):
        CREATE def __init__(self, config: EvaluatorsConfig):
            super().__init__(config)
            self.repo_id = "cais/mmlu"
            self.name = EvaluatorType.mmlu
            self.performance_breakdown = {} # first level maps subject to {num_correct: int, num_total: int}
            self.dataset = load_dataset(self.repo_id, subset="all", split="test")
            # rows have structure: {question: str, subject: str, choices: List[str], answer: int} (answer is index of correct choice)
            Include a processing step using dataset.map that applies a prompt to each row to get a prompt and expected response. Use a few-shot prompt.
            CREATE THIS PROMPT AND USE IT IN THE MAP. The shots should be the first 3 rows of the dataset.
            self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=False, prefetch_size=config.prefetch_size, num_workers=config.num_workers)

        CREATE def __call__(self, model: nn.Module) -> Dict[str, Dict[str, int]]:
            Iterate over each batch in the dataloader
            run batch through model
            get predictions
            evaluate against ground truth
            update performance_breakdown (adding to num_correct and num_total for the subject, adding subject to performance_breakdown if it doesn't exist).
            return performance_breakdown

```
3. Create GSM8KEvaluator class and file.
```aider
CREATE vorox/evaluators/gsm8k_evaluator.py:
    CREATE class GSM8KEvaluator(EvaluatorBase):
        MIRROR MMLUEvaluator
        repo_id = "openai/gsm8k"
        each row has structure: {question: str, answer: str}  answer is expected response where last line is "#### <answer>"
        Include a processing step using dataset.map that applies a prompt to each row to get a prompt and expected response.
        CREATE THIS PROMPT AND USE IT IN THE MAP. The shots should be the first 3 rows of the dataset.
        subset="main", split="test"
        Use "all" as the only subject in performance_breakdown.

```
4. Create GSMSymbolicEvaluator class and file.
```aider
CREATE vorox/evaluators/gsmsymbolic_evaluator.py:
    CREATE class GSMSymbolicEvaluator(EvaluatorBase):
        MIRROR MMLUEvaluator
        repo_id = "apple/GSM-Symbolic"
        each row has two relevant fields: {question: str, answer: str}  answer is expected response where last line is "#### <answer>"
        CREATE THIS PROMPT AND USE IT IN THE MAP. The shots should be the first 3 rows of the dataset.
        Include a processing step that applies a prompt to each row to get a prompt and expected response. Use the following template:
        ```
        As an expert problem solver, solve step by step the following mathematical questions.
        Q: <SHOT_1_QUESTION>
        A: Let's think step by step. <SHOT_1_ANSWER>. The final answer is <SHOT_1_FINAL_ANSWER>.
        .
        .
        .
        Q: <SHOT_3_QUESTION>
        A: Let's think step by step. <SHOT_3_ANSWER>. The final answer is <SHOT_3_FINAL_ANSWER>.

        Q: <TARGET_QUESTION>
        A: Let's think step by step.
        ```
        subset="p1", split="test"
        Use "all" as the only subject in performance_breakdown.
```
5. Create EvaluatorBuilder class with build method and file.
```aider
CREATE vorox/evaluators/builder.py:
    CREATE class EvaluatorBuilder:
        @staticmethod
        def build(cls, evaluators: List[EvaluatorType]) -> Evaluator:
            evaluators = []
            for eval_type in evaluators:
                if eval_type == EvaluatorType.mmlu:
                    evaluators.append(MMLUEvaluator())
                elif eval_type == EvaluatorType.gsm8k:
                    evaluators.append(GSM8KEvaluator())
                elif eval_type == EvaluatorType.gsm_symbolic:
                    evaluators.append(GSMSymbolicEvaluator())
                else:
                    print(f"Invalid evaluator type received: {eval_type}")

            return evaluators
```
6. Update configs/20M_test_model.yaml to include evaluators.
```aider
UPDATE vorox/configs/20M_test_model.yaml:
    ADD evaluators: [mmlu, gsm8k, gsm_symbolic]
```
7. Update train.py to include evaluators.
```aider
UPDATE scripts/train.py:
    ADD evaluator building using EvaluatorBuilder.
    Dont do anything with them yet, just build them.
```