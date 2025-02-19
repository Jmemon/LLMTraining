
### Downstream Evaluation vs Intrinsic Language Modeling (perplexity) Evaluation.
Downstream evaluation is a measure of how well a model performs on a specific task.
Intrinsic language modeling evaluation is about determining how well a model understands language.

Literally speaking, both will take the form of predicting the next token, what will vary will be the datasets and the metric.

The datasets will be structured around what the next tokens are. In downstream evaluation, they will be answers to a question (the prompt). The hope here is that correct answers imply an actual understanding of that question. The metric will be about the correctness of the answer (regardless of how many tokens that answer is).

In intrinsic language modeling, you won't necessarily structure the prompt so the next tokens are meant to test some kind of understanding (although they still can). Instead you will give it a very broad representation of some language domain, some style of language (Shakespeare, etc), or whatever else and see if it effectively predicts the next token. Here the metric will be about whether each token was correctly predicted, ie perplexity.

The intuition is that downstream evaluation should test more abstract understanding. Where perplexity evaluation is more about testing for the model having internalized linguistic structure. 

Concretely, what this might looks like for downstream evaluation is you'll give it a dataset of questions and answers. You'll feed it a question, it should complete the text with just the answer, then you see if it's correct. Eg a multiple choice question and you're expecting it to complete the text with "B".
```
In: 
    The order of operations or PEMDAS is
Out: 
    Parentheses Exponents Multiplication Division Addition Subtraction
```
```
In: 
    (1 + 1) × 2 = 
Out: 
    4
```
```
In:
    How many numbers are in the list 25, 26, ..., 100? (A) 75 (B) 76 (C) 22 (D) 23
    Answer: B
    Compute i+ i2+ i3+ ···+ i258+ i259. (A) -1 (B) 1 (C) i (D) -i
    Answer: A
    If 4 daps = 7 yaps, and 5 yaps = 3 baps, how many daps equal 42 baps?
    (A) 28 (B) 21 (C) 40 (D) 30
    Answer:
Out: 
    C
```

For intrinsic language modeling evaluation:
```
In: 
    Movie of Nelson Mandela’s life premieres in South Africa Nov. 04 - Stars Idris Elba and Naomie Harris attend the premiere of "Mandela: Long Walk to Freedom," based on the autobiography
Out:
    of anti-apartheid icon Nelson Mandela. Matthew Stock reports.
```

What does each contain? What is each supposed to measure?

MMLU
Hella-Swag
ARC-Challenge
ARC-Easy
PIQA
GPQA
WinoGrande