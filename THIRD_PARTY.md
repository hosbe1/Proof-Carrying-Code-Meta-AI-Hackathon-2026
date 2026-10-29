# Third-party code, data and models

Every dataset, demo source and model this project uses, with its source, its
license, what we use it for, and whether anything of it is committed here.
"Committed" means committed to this repository: nothing below is committed.
Datasets are downloaded by scripts, demo sources live in the separate demo
repository, and models are called through APIs.

Our own code is licensed under [AGPL-3.0](LICENSE); that license is out of
scope here.

| Source | License | What we use | Committed | Notice |
|---|---|---|---|---|
| HumanEvalFix | [MIT](https://github.com/openai/human-eval/blob/master/LICENSE) (data) | Evaluation of the whole pipeline and of the spec writer, in fixed and buggy versions, and training data for proof collection; the fixed test split of 40 functions never enters training. Data via the [humanevalfix task](https://github.com/bigcode-project/bigcode-evaluation-harness) of the BigCode evaluation harness | No — task IDs and a download script only | Copyright (c) OpenAI |
| HumanEval | [MIT](https://github.com/openai/human-eval/blob/master/LICENSE) | Training input for proof-task collection ([openai/human-eval](https://github.com/openai/human-eval)); HumanEvalFix test tasks are excluded | No — downloaded by script, never committed | Copyright (c) OpenAI |
| MBPP | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Training only, and only after the submission ([google-research/mbpp](https://github.com/google-research/google-research/tree/master/mbpp)) | No — downloaded by script, never committed | Attribution: Austin et al., Google |
| Vericoding | [MIT](https://github.com/vericoding/vericoding/blob/main/LICENSE) | Evaluation and training on the Dafny part (code and proof from a given spec, and proof-only tasks); the unlicensed VerifiedCogen part is excluded, not used and not trained on | No — task IDs and a download script only | Copyright (c) Vericoding contributors (full line in the linked LICENSE) |
| DafnyBench | [Apache-2.0](https://github.com/DafnyBench/DafnyBench/blob/main/LICENSE) | Not used before the submission; evaluation candidate afterwards ([DafnyBench/DafnyBench](https://github.com/DafnyBench/DafnyBench)) | No — nothing committed | Copyright (c) DafnyBench contributors (full line in the linked LICENSE) |
| BugsInPy | None — [no license file](https://github.com/soarsmu/BugsInPy) in the repository | Referenced by bug ID and commit only; fetched from the original projects at run time; never trained on | No — nothing committed | — |
| MBPP-DFY | [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html) | Local evaluation of translation and proof only, from a local snapshot of the Dafny translation of [MBPP](https://github.com/google-research/google-research/tree/master/mbpp); no upstream checkout used or committed | No — never committed | — |
| TheAlgorithms/Python | [MIT](https://github.com/TheAlgorithms/Python/blob/master/LICENSE.md) | Demo: `strings/palindrome.py` and `strings/z_function.py` copied into the demo repository with the notice | No in this repository — the copies with the notice live in the separate demo repository | Copyright (c) TheAlgorithms |
| Muse Spark | Proprietary [commercial terms](https://www.meta.ai/legal/commercial-terms) — no license file | Intent, spec and AI fallback translation via the Meta Model API (`muse-spark-1.3`); proof stage 2; standard mode, and contributor mode (`muse-spark-1.3-contributor`) for public data in the Meta environment | No — API only, nothing committed | — |
| Muse Wonder | Proprietary [commercial terms](https://www.meta.ai/legal/commercial-terms) — no license file | Proof stage 3 only via the Meta Model API (`muse-wonder-1.0`), for functions of at most 40 source lines; used sparingly, no contributor mode | No — API only, nothing committed | — |
| Muse Glimmer | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) (base-model license, needed for the HF adapter release) | Proof stage 1 everywhere: evaluation, demo and bug hunt; the trained LoRA adapter release on Hugging Face carries its results and the base-model license | No — the adapter release lives on Hugging Face, nothing committed here | Copyright (c) base-model holders (full line in the linked base-model license) |

Notes on the notices: MIT, Apache-2.0 and CC BY 4.0 require keeping the
copyright or attribution notice when the material is redistributed. The Notice
column names the holder or attribution; the full copyright line is in the
linked license file. This
repository redistributes none of the material above, so no notice is vendored
here; the demo repository carries TheAlgorithms' MIT notice with its two
copied files.
