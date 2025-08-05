# Combining Self-RAG with Divide-and-Conquer for LM-KBC 2025

**2nd Place Solution** on the LM-KBC 2025 Challenge Hidden Test Leaderboard

This repository contains our hybrid approach that combines Self-Retrieval-Augmented Generation (Self-RAG) with Divide-and-Conquer strategies for language model-based knowledge base construction.

## Table of Contents
- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Details](#model-details)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Overview

Our system addresses the LM-KBC 2025 challenge of constructing disambiguated knowledge bases using only a fixed language model (Qwen3-8B) without fine-tuning or external retrieval. We developed a hybrid approach that:

- **Self-RAG Pipeline**: Handles general relations through description-first knowledge activation
- **Divide-and-Conquer Pipeline**: Specializes in high-cardinality enumeration for `awardWonBy` relations
- **Specification-driven Prompting**: Eliminates post-processing errors through strict output formatting

## Key Results

| Metric | Baseline | Our Method | Improvement |
|--------|----------|------------|-------------|
| **Macro F1** | 0.2116 | **0.4052** | **+91.5%** |
| Precision | 0.2272 | 0.5234 | +0.2962 |
| Recall | 0.4348 | 0.4590 | +0.0242 |

### Per-Relation Performance

| Relation | Baseline F1 | Our F1 | Δ |
|----------|-------------|--------|---|
| `awardWonBy` | 0.1170 | 0.1759 | +0.0589 |
| `companyTradesAtStockExchange` | 0.1670 | **0.5057** | **+0.3387** |
| `countryLandBordersCountry` | 0.7025 | **0.8649** | **+0.1624** |
| `hasArea` | 0.2400 | 0.3100 | +0.0700 |
| `hasCapacity` | 0.0400 | 0.1100 | +0.0700 |
| `personHasCityOfDeath` | 0.0800 | **0.4100** | **+0.3300** |

## Architecture

```
Input: (Subject Entity, Relation)
              │
              ▼
         Relation Type?
              │
    ┌─────────┴─────────┐
    ▼                   ▼
awardWonBy         Other Relations
    │                   │
    ▼                   ▼
Divide & Conquer    Self-RAG Pipeline
Pipeline                │
    │                   ▼
    │              1. Description Generation
    │              2. Targeted Extraction  
    │              3. Response Processing
    │                   │
    └───────────────────┴───────► Entity List
```

### Self-RAG Pipeline (5 Relations)
1. **Description Generation**: Create relation-specific entity descriptions
2. **Targeted Extraction**: Use descriptions to guide precise knowledge extraction
3. **Response Processing**: Minimal cleaning with strict validation

### Divide-and-Conquer Pipeline (awardWonBy)
1. **Query Decomposition**: Split into temporal, geographic, and direct queries
2. **Candidate Aggregation**: Collect results from multiple constrained subqueries
3. **Name Validation**: Apply strict filters to ensure valid recipient names

## Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/LM-KBC-ISWC-2025.git
cd LM-KBC-ISWC-2025

# Install dependencies
pip install -r requirements.txt

# Ensure you have access to Qwen3-8B model
# The model will be automatically downloaded from Hugging Face
```

## Usage

### Quick Start

```bash
# Run Self-RAG model on all non-awardWonBy relations
python baseline.py --config_file configs/self_rag_config.yaml --input_file data/val.jsonl

# Run Divide-and-Conquer model for awardWonBy
python main.py --config_file configs/divide_conquer_config.yaml --input_file data/val.jsonl
```

### Evaluation

```bash
# Evaluate on validation set
python evaluate.py --predictions results/predictions.jsonl --ground_truth data/val.jsonl
```

## Configuration

### Self-RAG Configuration (`configs/self_rag_config.yaml`)

```yaml
model: "self_rag"
llm_path: "Qwen/Qwen3-8B"
prompt_templates_file: "prompt_templates/question_prompts.csv"
max_new_tokens: 32768
use_quantization: false
few_shot: 5
train_data_file: "data/train.jsonl"

# Self-RAG specific settings
use_description: true
description_max_tokens: 150

# Logging
save_logs: true
log_dir: "experiment_logs"
```

### Divide-and-Conquer Configuration (`configs/divide_conquer_config.yaml`)

```yaml
model: "improved_divide_conquer"
llm_path: "Qwen/Qwen3-8B"
max_new_tokens: 32768
few_shot: 0  # Disabled to avoid interference

# Divide-and-conquer settings
use_divide_conquer: ["awardWonBy"]
max_query_attempts: 3

# Logging
save_logs: true
log_dir: "experiment_logs"
```

## Model Details

### Self-RAG Implementation

Located in `models/self_rag_model.py`:

- **Description Generation**: Relation-specific prompts to activate relevant knowledge
- **Targeted Extraction**: Context-conditioned queries with strict output specifications
- **Response Processing**: Minimal cleaning with uncertainty handling

Key features:
- Temperature: 0.1 for consistent outputs
- Strict format specifications: "Answer with one number only", "Names only, comma-separated"
- Proactive uncertainty handling: "If uncertain, answer 'none'"

### Divide-and-Conquer Implementation

Located in `models/improved_divide_conquer_model.py`:

- **Temporal Slicing**: 8 decade-based categories (1950s-2020s)
- **Geographic Slicing**: 9 nationality categories + "other"
- **Direct Enumeration**: 5 backup query formulations
- **Name Validation**: Multi-stage filtering with strict criteria

Key features:
- Comprehensive coverage through systematic decomposition
- Aggressive name filtering to reduce false positives
- Candidate aggregation with deduplication

## Project Structure

```
LM-KBC-ISWC-2025/
├── configs/                    # Configuration files
│   ├── self_rag_config.yaml
│   └── divide_conquer_config.yaml
├── models/                     # Model implementations
│   ├── baseline_qwen_3_model.py    # Base model (from competition)
│   ├── self_rag_model.py           # Our Self-RAG implementation
│   └── divide_conquer_model.py  # Our D&C implementation
├── data/                       # Dataset files
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── prompt_templates/           # Prompt templates
├── experiment_logs/            # Generated logs and statistics
├── results/                    # Prediction outputs
├── main.py                     # Main execution script
├── run_hybrid_system.py        # Hybrid system runner
├── evaluate.py                 # Evaluation utilities
└── requirements.txt            # Dependencies
```

## Key Innovations

1. **Hybrid Architecture**: Strategic allocation of computational resources based on relation complexity
2. **Specification-driven Prompting**: 0% formatting failures through careful prompt design
3. **Systematic Error Analysis**: 96% of errors stem from knowledge gaps, not technical issues
4. **Strict Name Validation**: Multi-stage filtering for high-quality candidate extraction

## Computational Analysis

| Method | Relations | Baseline Time | Our Time | Overhead |
|--------|-----------|---------------|----------|----------|
| Self-RAG | Non-awardWonBy | 2h 12m | 2h 28m | 1.11× |
| Divide & Conquer | awardWonBy | 1h 8m | 4h 22m | 3.85× |
| **Hybrid System** | **All** | **3h 21m** | **6h 49m** | **2.04×** |

The strategic 2.04× computational investment yields 91.5% F1 improvement.

## Citation

```bibtex
@inproceedings{he2025combining,
  title={Combining Self-Retrieval-Augmented Generation with Divide-and-Conquer for Language Model-based Knowledge Base Construction},
  author={He, Jingbo and Razniewski, Simon},
  booktitle={Joint proceedings of KBC-LM and LM-KBC @ ISWC 2025},
  year={2025}
}
```

## Acknowledgments

- **LM-KBC 2025 Challenge Organizers** for the dataset and evaluation framework
- **Competition Template**: This work builds upon the [official LM-KBC 2025 template](https://github.com/lm-kbc/dataset2025)
- **Qwen Team** for the Qwen3-8B base model
- **TU Dresden & ScaDS.AI** for computational resources

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

## Contact

- **Jingbo He**: jingbo.he@mailbox.tu-dresden.de
- **Simon Razniewski**: simon.razniewski@tu-dresden.de
- **Paper**: [arXiv link will be added]
- **Competition**: https://lm-kbc.github.io/challenge2025/
