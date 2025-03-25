
# Evaluators

## High-Level Objective

## Mid-Level Objectives

## Implementation Notes

## Context
### Beginning Context
vorox/config.py (read-only)

### Ending Context
vorox/config.py (read-only)

## Low-Level Tasks
1. Restructure config.py to have a new section for evaluators.
```aider
UPDATE vorox/config.py:
    RENAME all config class names to have "Config" suffix

    CREATE pydantic EvaluatorConfig(BaseModel):
        evaluators: List[EvaluatorType]
        batch_size: int
        num_workers: int
        prefetch_size: int

    UPDATE Data:
        DELETE evaluators: List[EvaluatorType]

    UPDATE Config (rename to RunConfig):
        ADD eval: EvaluatorsConfig
```
2. Create EvaluatorBase class and file.
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
```
4. Create GSMSymbolicEvaluator class and file.
```aider
CREATE vorox/evaluators/gsmsymbolic_evaluator.py:
    CREATE class GSMSymbolicEvaluator(EvaluatorBase):
```
5. Create ArcAGIEvaluator class and file.
```aider
CREATE vorox/evaluators/arc_agi_evaluator.py:
    CREATE class ArcAGIEvaluator(EvaluatorBase):
```
```
6. Create EvaluatorBuilder class with build method and file.
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
                elif eval_type == EvaluatorType.arc_agi:
                    evaluators.append(ArcAGIEvaluator())
                else:
                    print(f"Invalid evaluator type received: {eval_type}")

            return evaluators
```
