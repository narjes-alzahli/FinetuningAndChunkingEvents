# Fine-Tuning and Chunking of Event Representations in the Brain using Repeated Naturalistic Stimuli

Many experiences are not unique but tend to be repeated: we might watch the same movie more than once and listen to the same song on repeat. How does the brain modify its representations of events when experiences are repeated? The "fine-tuning" hypothesis predicts that brain regions become more sensitive to details with repeated exposure to a stimulus, leading to more granular neural events. The "chunking" view suggests that brain regions integrate events into more generalized representations with repeated exposure, leading to coarser neural events. To test these accounts, we analyzed data from 30 human participants who underwent functional magnetic resonance imaging (fMRI) while watching three 90-second clips from "The Grand Budapest Hotel" six times each. We used hidden Markov models applied to searchlights across the brain to identify event timescales for each viewing and tested how these event timescales change with repeated viewings. Most brain regions exhibited stability in their preferred timescale across repeated viewings. A smaller set of regions showed flexible event representations that became more or less granular with clip repetitions, showing evidence for both “fine-tuning” and “chunking” hypotheses in different regions. Notably, in superior temporal gyrus and hippocampus, greater fine-tuning of event representations predicted more detailed memory recall. These results highlight the importance of treating event representations in the brain not as inherent or fixed, but flexible with experience.

---

## Features

- **slurm_main.py**: Implements the main pipeline for running permutations on searchlights in the brain to determine their optimal event timescale for each clip viewing.
- **slurm_utils.py**: Helper functions and utilities used by `slurm_main.py`.
- **slurm_jobs.sh**: A shell script for submitting and managing jobs to execute the analysis pipeline on searchlights.