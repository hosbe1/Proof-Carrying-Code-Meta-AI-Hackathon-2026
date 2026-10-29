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
| [HumanEvalFix](https://huggingface.co/datasets/bigcode/humanevalpack) | MIT (no separate dataset LICENSE file; the [dataset card](https://huggingface.co/datasets/bigcode/humanevalpack/blob/main/README.md) declares license: mit); authors' code [MIT LICENSE](https://github.com/bigcode-project/octopack/blob/main/LICENSE) ([octopack](https://github.com/bigcode-project/octopack)) | Evaluation of the whole pipeline and of the spec writer, in fixed and buggy versions, and training data for proof collection; the fixed test split of 40 functions never enters training. Data via the HumanEvalPack dataset; task IDs and a download script only | No — task IDs and a download script only | Copyright (c) 2023 Muennighoff (Octopack LICENSE); HumanEval tasks carry Copyright (c) OpenAI as applicable |
| [HumanEval](https://github.com/openai/human-eval) | [MIT LICENSE](https://github.com/openai/human-eval/blob/master/LICENSE) | Training input for proof-task collection; HumanEvalFix test tasks are excluded. Downloaded by script | No — downloaded by script, never committed | Copyright (c) OpenAI (no year given upstream) |
| [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) | CC BY 4.0 license: the upstream [repository README](https://github.com/google-research/google-research/blob/master/README.md) declares all datasets CC BY 4.0; canonical [CC BY 4.0 legal text](https://creativecommons.org/licenses/by/4.0/legalcode) | Training only, and only after the submission. Downloaded by script | No — downloaded by script, never committed | Attribution: Austin et al., 2021, per [mbpp README](https://github.com/google-research/google-research/blob/master/mbpp/README.md) |
| [Vericoding](https://github.com/Beneficial-AI-Foundation/vericoding-benchmark) | [MIT LICENSE](https://github.com/Beneficial-AI-Foundation/vericoding-benchmark/blob/main/LICENSE) | Evaluation and training on the Dafny part (code and proof from a given spec, and proof-only tasks); the unlicensed VerifiedCogen part is excluded, not used and not trained on. Component tasks keep their own licenses per the upstream README | No — task IDs and a download script only | Copyright (c) 2025 Beneficial AI Foundation (from the linked LICENSE) |
| [DafnyBench](https://github.com/sun-wendy/DafnyBench) | [Apache-2.0 LICENSE](https://github.com/sun-wendy/DafnyBench/blob/main/LICENSE) | Not used before the submission; evaluation candidate afterwards | No — nothing committed | Upstream LICENSE contains only the unfilled Apache appendix Copyright [yyyy] [name of copyright owner] with no project-specific notice; attribution follows the dataset authors as cited in the upstream README/citation |
| [BugsInPy](https://github.com/soarsmu/BugsInPy) | None — [no license file](https://github.com/soarsmu/BugsInPy) in the repository | Referenced by bug ID and commit only; fetched from the original projects at run time; never trained on | No — nothing committed | — |
| [MBPP-DFY](https://github.com/Mondego/vRAG/tree/master/MBPP-DFY-50) | [GPL-3.0 LICENSE](https://github.com/Mondego/vRAG/blob/master/LICENSE) (the upstream README lists MBPP-DFY-50 and MBPP-DFY-153 and declares GPL-3.0) | Local evaluation of translation and proof only, from a local snapshot of the Dafny translation ([MBPP-DFY-153](https://github.com/Mondego/vRAG/tree/master/MBPP-DFY-153) lives in the same repository); no upstream checkout used or committed | No — never committed | Upstream LICENSE is the GPL-3.0 license text; its FSF copyright covers the license text, not dataset authorship |
| [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) | [MIT LICENSE](https://github.com/TheAlgorithms/Python/blob/master/LICENSE.md) | Demo: `strings/palindrome.py` and `strings/z_function.py` copied into the demo repository with the notice | No in this repository — the copies with the notice live in the separate demo repository | Copyright (c) 2016-2022 TheAlgorithms and contributors |
| [Muse Spark](https://dev.meta.ai/docs/pricing-rate-limits) | Proprietary — no license file; [Model API Terms of Service](https://dev.meta.ai/legal/terms-of-service) (last updated October 9, 2026) and [Acceptable Use Policy](https://dev.meta.ai/legal/acceptable-use-policy) apply | Intent, spec and AI fallback translation via the Model API (`muse-spark-1.3`); proof stage 2; standard mode, and contributor mode (`muse-spark-1.3-contributor`) for public data in the Meta environment | No — API only, nothing committed | — |
| [Muse Wonder](https://dev.meta.ai/docs/pricing-rate-limits) | Proprietary — no license file; [Model API Terms of Service](https://dev.meta.ai/legal/terms-of-service) (last updated October 9, 2026) and [Acceptable Use Policy](https://dev.meta.ai/legal/acceptable-use-policy) apply | Proof stage 3 only via the Model API (`muse-wonder-1.0`), for functions of at most 40 source lines; used sparingly, no contributor mode | No — API only, nothing committed | — |
| [Muse Glimmer](https://huggingface.co/meta-models/Muse-Glimmer-30B) | [Apache-2.0 base-model LICENSE](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/LICENSE), needed for the HF adapter release (license also confirmed on the [model page](https://dev.meta.ai/models/muse-glimmer)) | Proof stage 1 everywhere: evaluation, demo and bug hunt; no trained adapter published yet, so no adapter URL is listed | No — the adapter release lives on Hugging Face, nothing committed here | Upstream LICENSE contains only the unfilled Apache appendix with no project-specific copyright notice; author per the official model card: Meta Superintelligence Lab |

Notes on the notices: MIT, Apache-2.0 and CC BY 4.0 require keeping the
copyright or attribution notice when the material is redistributed. The Notice
column names the holder or attribution; where upstream provides a filled
notice the full line is in the linked license file. Two upstream Apache
LICENSE files (DafnyBench, Muse-Glimmer-30B) contain only the unfilled
appendix with no project-specific notice, which the table states instead of
inventing one. HumanEvalFix has no separate dataset LICENSE file: the linked
dataset card declares MIT and the linked Octopack LICENSE carries the code
notice. MBPP has no per-dataset LICENSE file: the linked repository README
declares datasets CC BY 4.0 and the table links the canonical legal text.
Muse Spark and Muse Wonder are proprietary API services with no license file;
the table cites their Model API terms. This repository redistributes none of
the material above, so no notice is vendored here; the demo repository
carries TheAlgorithms' MIT notice with its two copied files.
