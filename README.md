# Title: Citation-Enhanced Generation for LLM-based Chatbots

![Main Figure](./pictures/method.png)  

## Introduction

This repository provides the implementation of our paper:  
**Citation-Enhanced Generation for LLM-based Chatbots**  
**Authors:** [Weitao Li, Junkai Li, Weizhi Ma, Yang Liu]  
Published in [ACL 2024]

[Paper Link](https://arxiv.org/abs/2402.16063)

### Abstract

Large language models (LLMs) exhibit powerful general intelligence across diverse scenarios, including their integration into chatbots. However, a vital challenge of LLM-based chatbots is that they may produce hallucinated content in responses, which significantly limits their applicability. Various efforts have been made to alleviate hallucination, such as retrieval augmented generation and reinforcement learning with human feedback, but most of them require additional training and data annotation. In this paper, we propose a novel post-hoc **C**itation-**E**nhanced **G**eneration (**CEG**) approach combined with retrieval argumentation. Unlike previous studies that focus on preventing hallucinations during generation, our method addresses this issue in a post-hoc way. It incorporates a retrieval module to search for supporting documents relevant to the generated content, and employs a natural language inference-based citation generation module. Once the statements in the generated content lack of reference, our model can regenerate responses until all statements are supported by citations. Note that our method is a training-free plug-and-play plugin that is capable of various LLMs. Experiments on various hallucination-related datasets show our framework outperforms state-of-the-art methods in both hallucination detection and response regeneration on three benchmarks.

## Installation

To set up the environment, run:

```bash
pip install -r requirements.txt
```

Ensure you have the required dependencies installed.

## Usage

### Data Preparation
1. Download and preprocess the dataset.
2. Construct the **Document Graph (DocGraph)** using:

```bash
python build_doc_graph.py --input data/raw_documents.json --output data/doc_graph.pkl
```

### Running the Model

Train the retrieval-augmented generation model with:

```bash
python train.py --config configs/config.yaml
```

### Evaluation

Evaluate the model performance using:

```bash
python evaluate.py --checkpoint path/to/model.ckpt --dataset data/test.json
```

## Experiments

We conduct experiments on multiple document collections, comparing **Graph-of-Docs** with standard RAG baselines. Our findings demonstrate:
- Improved retrieval quality due to the structured document graph.
- Enhanced generation fidelity by leveraging fine-grained document relationships.
- Better performance on factual consistency metrics.

## Results

| Model | Retrieval Accuracy | Generation Quality (BLEU/ROUGE) |
|--------|------------------|----------------------------|
| Standard RAG | XX.X | XX.X |
| **Graph-of-Docs (Ours)** | **XX.X** | **XX.X** |

## Citation

If you use this code, please cite:

```
@article{li2024citation,
  title={Citation-Enhanced Generation for LLM-based Chatbots},
  author={Li, Weitao and Li, Junkai and Ma, Weizhi and Liu, Yang},
  journal={arXiv preprint arXiv:2402.16063},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

