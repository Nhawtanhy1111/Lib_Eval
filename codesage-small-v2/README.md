---
license: apache-2.0
datasets:
- bigcode/the-stack-dedup
- bigcode/the-stack-v2

library_name: transformers
language:
- code
---

## CodeSage-Small-v2


### Updates
* 1. <span style="color:blue">We are excited to announce the release of the CodeSage V2 model family with largely improved performance!</span> <strong>Please check out our [blogpost](https://code-representation-learning.github.io/codesage-v2.html) for more details</strong>.
* 2. In addition to retrieval performance boost, V2 models also support flexible embedding sizes (thanks to Matryoshka Representation Learning (MRL)).
* 3. You can access CodeSage v2 models through both HF and SentenceTransformer. 

### Model description
CodeSage is a family of open code embedding models with an encoder architecture that supports a wide range of source code understanding tasks. It was initially introduced in the paper:

[Code Representation Learning At Scale by Dejiao Zhang*, Wasi Uddin Ahmad*, et al.](https://arxiv.org/abs/2402.01935)  

For this V2 model, we enhanced semantic search performance by improving the quality of the contrastive learning data through [consistency filtering](https://arxiv.org/abs/2209.11755). Starting from the pretrained checkpoint (trained with both Masked Language Modeling (MLM) and deobfuscation [Section 3.1](https://arxiv.org/abs/2402.01935)) from our V1 model training (Zhang et al., 2023), we applied contrastive learning with the filtered data. Unlike the V1 model, we extracted the initial set of (text, code) pairs—specifically, summaries and function/class bodies—from [The Stack V2](https://huggingface.co/datasets/bigcode/the-stack-v2) data instead of using the [V1](https://huggingface.co/datasets/bigcode/the-stack-dedup) data. We employed simple rule-based filtering as detailed in our previous work. We then applied consistency filtering to further refine the data. While using The Stack V2 resulted in minor performance boosts on downstream tasks, the majority of the performance improvements came from the consistency filtering.

### Model Performance

#### 1.Code2Code Search
| Model Name          | # Params | Embd Dim | Python | Java  | JS    | TS     | C#     | C      | Ruby   | PhP    | GO     | AVG    |
|---------------------|----------|----------|--------|-------|-------|--------|--------|--------|--------|--------|--------|--------|
| OpenAI-Code-01      | NA       | 3072     | 21.92  | 8.90  | 4.90  | 5.70   | 3.15   | 11.58  | 26.25  | 16.60  | 9.40   | 12.04  |
| OpenAI-Ada-002      | NA       | 1536     | 35.91  | 25.13 | 19.01 | 21.86  | 10.17  | 29.15  | 40.85  | 40.47  | 23.43  | 27.33  |
| OpenAI-Text-3-Small | NA       | 1536     | 25.18  | 12.61 | 8.00  | 9.44   | 5.46   | 15.86  | 30.70  | 23.33  | 11.20  | 15.57  |
| OpenAI-Text-3-Large | NA       | 3072     | 40.57  | 25.33 | 20.09 | 22.00  | 11.84  | 31.90  | 42.54  | 41.84  | 21.75  | 28.65  |
| CodeSage-Small      | 130M     | 1024     | 36.31  | 23.97 | 26.60 | 29.90  | 11.84  | 22.84  | 29.06  | 34.64  | 19.56  | 26.08  |
| CodeSage-Base       | 356M     | 1024     | 47.52  | 22.84 | 28.70 | 31.95  | 13.37  | 30.99  | 44.86  | 51.13  | 25.15  | 32.95  |
| CodeSage-Large      | 1.3B     | 2048     | 46.70  | 33.13 | 37.16 | 41.18  | 16.81  | 32.89  | 54.12  | 52.13  | 32.48  | 38.51  |
| CodeSage-v2-Small   | 130M     | 1024     | 45.60  | 33.65 | 39.96 | 47.78  | 19.19  | 30.55  | 40.12  | 55.39  | 30.96  | 38.13  |
| CodeSage-v2-Base    | 356M     | 1024     | 55.86  | 42.89 | 45.29 | 54.58  | 23.90  | 38.52  | 56.02  | 64.56  | 42.88  | 47.17  |
| CodeSage-v2-Large   | 1.3B     | 2048     | 61.11  | 47.09 | 51.18 | 60.67  | 28.04  | 43.40  | 60.74  | 67.87  | 43.86  | 51.55  |


#### 2. NL2Code Search
| Model Name          | # Params | CoSQA | AdvTest | Python | Java  | JS    | PhP    | GO     | Ruby   | Avg    |
|---------------------|----------|-------|---------|--------|-------|-------|--------|--------|--------|--------|
| OpenAI-Code-01      | NA       | 52.20 | 36.03   | 63.13  | 67.85 | 62.30 | 57.47  | 85.22  | 69.28  | 61.69  |
| OpenAI-Ada-002      | NA       | 44.23 | 38.08   | 68.02  | 71.49 | 67.50 | 60.62  | 85.63  | 74.20  | 63.72  |
| OpenAI-Text-3-Small | NA       | 52.48 | 34.10   | 62.62  | 65.87 | 60.28 | 54.85  | 81.96  | 67.57  | 59.97  |
| OpenAI-Text-3-Large | NA       | 55.21 | 46.83   | 70.81  | 72.89 | 68.12 | 59.58  | 87.60  | 75.22  | 67.03  |
| CodeSage-Small      | 130M     | 49.93 | 41.05   | 64.26  | 63.19 | 59.87 | 54.65  | 77.60  | 63.18  | 59.22  |
| CodeSage-Base       | 356M     | 48.50 | 48.87   | 67.81  | 68.00 | 66.87 | 58.13  | 83.17  | 68.00  | 63.67  |
| CodeSage-Large      | 1.3B     | 47.49 | 52.35   | 70.64  | 70.20 | 69.54 | 61.31  | 83.60  | 71.88  | 65.88  |
| CodeSage-v2-Small   | 130M     | 52.39 | 47.28   | 68.79  | 68.13 | 65.77 | 60.20  | 80.26  | 72.46  | 64.41  |
| CodeSage-v2-Base    | 356M     | 50.74 | 52.00   | 70.46  | 70.89 | 69.61 | 62.81  | 82.37  | 73.71  | 66.57  |
| CodeSage-v2-Large   | 1.3B     | 53.18 | 56.31   | 74.18  | 72.33 | 72.49 | 65.26  | 84.67  | 76.61  | 69.38  |







### Training Data
This pretrained checkpoint is the same as those used by our V1 model ([codesage/codesage-small](https://huggingface.co/codesage/codesage-small), which is trained on [The Stack](https://huggingface.co/datasets/bigcode/the-stack-dedup) data. The constative learning data are extracted from [The Stack V2](https://huggingface.co/datasets/bigcode/the-stack-v2). Same as our V1 model, we supported nine languages as follows: c, c-sharp, go, java, javascript, typescript, php, python, ruby.

### How to Use
This checkpoint consists of an encoder (130M model), which can be used to extract code embeddings of 1024 dimension. 

1. Accessing CodeSage via HuggingFace: it can be easily loaded using the AutoModel functionality and employs the [Starcoder Tokenizer](https://arxiv.org/pdf/2305.06161.pdf).
   
```
from transformers import AutoModel, AutoTokenizer

checkpoint = "codesage/codesage-small-v2"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

# Note: CodeSage requires adding eos token at the end of each tokenized sequence 

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, add_eos_token=True)

model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

inputs = tokenizer.encode("def print_hello_world():\tprint('Hello World!')", return_tensors="pt").to(device)
embedding = model(inputs)[0]
```

2. Accessing CodeSage via SentenceTransformer
```
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("codesage/codesage-small-v2", trust_remote_code=True)
```

### BibTeX entry and citation info
```
@inproceedings{
    zhang2024code,
    title={{CODE} {REPRESENTATION} {LEARNING} {AT} {SCALE}},
    author={Dejiao Zhang and Wasi Uddin Ahmad and Ming Tan and Hantian Ding and Ramesh Nallapati and Dan Roth and Xiaofei Ma and Bing Xiang},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=vfzRRjumpX}
}
```