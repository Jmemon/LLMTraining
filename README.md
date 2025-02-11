# LLMTraining
**Phase 1:** Build the infrastructure to train and perform inference with 10M parameter model. Enables fast iteration so we can learn the system well. **Goal** is the training setup. Not about arch experimentation or anything.
1. what is the OLMoE train loop? Know it, implement it (from scratch).
	1. Evaluation?
	2. data etl?
	3. dataset? (DataComp-LM + Dolma)
	4. inference? – how to have it reason?
	5. Effective Artifact storage? (just use hf?)
	6. Log everything possible for each training run – to ensure, make the kickoff script ask user for anything needed to ensure this:
		- hardware
		- architecture
		- optimizer
		- initialization scheme (rand/transfer/etc)
		- hyperparameters
		- data subset
		- data order
		- loss
		- eval metrics
		- per-device timeline of compute/memory usage (percent usage follows)
		- time per batch, time per update
		- What else?
2. Use a dummy model (ie has necessary functions, but does nothing) to run through loop so we know it works.
3. Create a 10M param arch – keep it relatively simple for now
4. Complete a bunch of tasks (all connected to basically training a lot)
	- hyper-paremeter optimization
	- How different data subsets affect evals.
	- How data order affects evals – how to do curriculum learning? – or empirically validate curriculum learning?
	- some interpretability?
	- comparing training runs
	- metrics
	- connecting metric curve occurrences to what was being fed to the model at that time

*Vision:* Play to such a point that I build some kind of mental model for training where scaling up the model size is natural.

*Once here move to phase 2.*
