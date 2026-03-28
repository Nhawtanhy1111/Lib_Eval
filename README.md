# LibEvolutionEval: A Benchmark and Study for Version-Specific Code Generation

- **🌐 Project Website**: [https://lib-evolution-eval.github.io/](https://lib-evolution-eval.github.io/) 
- **Venue**: NAACL 2025
- **Presentation**: Oral Presentation
- **📄 Paper**: [https://aclanthology.org/2025.naacl-long.348/](https://aclanthology.org/2025.naacl-long.348/)

## Overview

Recent advancements in code completion models have primarily focused on either single-file contexts or entire repositories, overlooking the critical challenge of fast-evolving public libraries. As libraries introduce, deprecate, or modify APIs across versions, a model’s performance can vary significantly based on the year and version of the library in use.

We propose LIBEVOLUTIONEVAL, a benchmark and detailed study that explicitly evaluates code language models under version-specific code completion tasks. Spanning multiple years and covering popular Python libraries, it provides both real-world GitHub-based examples and controlled, documentation-based tasks. Our experiments with popular code LLMs and embedding models reveal that addressing library evolution demands version-aware context. While retrieval of version-specific documentation can improve code completions, it does not fully resolve version-dependent biases in the models, indicating the need for specialized training techniques to handle rapid library changes.

## Dataset

LibEvolutionEval provides a comprehensive benchmark for evaluating code completion models on version-specific tasks across multiple Python libraries. The dataset includes:

- Version-specific code completion
- Real-world GitHub examples
- Documentation-based synthetic tasks
- Library evolution tracking

## Dataset Statistics

| Feature | PyTorch | Matplotlib |
|---------|---------|------------|
| # API Documentations | 29.4K | 35.6K |
| # Eval Examples | 20.8K | 9.6K |

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended for model inference)

### Dependencies

Install the required dependencies using conda:

```bash
conda env create -f requirements.yml
conda activate starcoder
```

### Dataset Setup

To set up the complete dataset, extract all archives:

```bash
# Create directories if they don't exist
mkdir -p data/eval_examples
mkdir -p data/package_apis

# Extract all archives
tar -xzf data/pytorch_eval_examples.tar.gz -C data/eval_examples/
tar -xzf data/matplotlib_eval_examples.tar.gz -C data/eval_examples/
tar -xzf data/package_apis.tar.gz -C data/package_apis
```

After extraction, your data structure should look like:
```
data/
├── eval_examples/
│   ├── torch_direct_api/
│   ├── torch_hard_introduced_or_deprecated/
│   ├── torch_indirect_api/
│   ├── torch_synth_1000/
│   ├── torch_synth_deprecated/
│   ├── matplotlib_direct_api/
│   ├── matplotlib_indirect_api/
│   ├── matplotlib_synth_1000/
│   ├── matplotlib_synth_deprecated/
│   └── matplotlib_synth_introduced/
└── package_apis/
```

## Usage

Our evaluation consists of two steps: **generation** and **metrics calculation**.

### Step 1: Generation

Generate code completions using the evaluation pipeline. This step produces model outputs for the given tasks.

#### Basic Generation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --task eval_examples \
    --mode generate \
    --model starcoder2-3b \
    --base_model starcoder2-3b \
    --source torch_direct_api \
    --api_version v_2_0_0 \
    --task_version v_2_0_0 \
    --setting_desc_variable results
```

#### Additional Options

- `--debug` - Process only first 10 samples for testing
- `--setting_desc_variable` - Custom results directory

### Step 2: Metrics Calculation

After generating completions, calculate evaluation metrics (Edit Similarity, Exact Match, etc.).

#### Basic Scoring

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --task eval_examples \
    --mode score \
    --model starcoder2-3b \
    --base_model starcoder2-3b \
    --source torch_direct_api \
    --api_version v_2_0_0 \
    --task_version v_2_0_0 \
    --setting_desc_variable results
```

### Automated Evaluation

For convenience, use the provided script to run evaluations across multiple configurations:

```bash
# Run all configured evaluations
./run.sh

# Run in debug mode (first 10 samples only)
./run.sh --debug
```

## Authors

Sachit Kuhar, Wasi Uddin Ahmad, Zijian Wang, Nihal Jain, Haifeng Qian, Baishakhi Ray, Murali Krishna Ramanathan, Xiaofei Ma, Anoop Deoras

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{kuhar-etal-2025-libevolutioneval,
    title = "{L}ib{E}volution{E}val: A Benchmark and Study for Version-Specific Code Generation",
    author = "Kuhar, Sachit  and
      Ahmad, Wasi Uddin  and
      Wang, Zijian  and
      Jain, Nihal  and
      Qian, Haifeng  and
      Ray, Baishakhi  and
      Ramanathan, Murali Krishna  and
      Ma, Xiaofei  and
      Deoras, Anoop",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.348/",
    pages = "6826--6840",
    ISBN = "979-8-89176-189-6",
    abstract = "Recent advancements in code completion models have primarily focused on local file contexts. However, these studies do not fully capture the complexity of real-world software development, which often requires the use of rapidly-evolving public libraries. To address this gap, we introduce LibEvolutionEval, a comprehensive study that emphasizes the need to understand library evolution to perform accurate in-line code completions. LibEvolutionEvaloffers a version-specific code-completion task across eight libraries as they evolve over the years, along with an in-depth analysis of the evolution of two widely used and well-maintained public libraries: PyTorch and Matplotlib. We evaluate several popular models and find that public library evolution significantly affects their performance. To mitigate this, we explored how retrieving version-specific library documentation and prompt-based techniques can enhance model capability in dealing with these fast-evolving packages. This suggests a promising path forward for better handling fast-evolving libraries. Our tasks will be made publicly available upon acceptance."
}
```


## Acknowledgments

This work was conducted at AWS AI Labs. We thank the research community for their foundational work in code completion evaluation and the open source community for providing the libraries and tools that made this research possible. This code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests for code changes are not being accepted in order to maintain the code as it was used in the paper.
