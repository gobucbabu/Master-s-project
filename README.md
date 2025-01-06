# Film Review Autocompletion: Pre-trained vs Custom Language Models

A comparative study investigating the performance and resource efficiency of fine-tuned DistilGPT-2 versus a custom transformer model for film review autocompletion.

## Project Overview
This project explores whether custom language models remain viable in an era of readily available pre-trained models, focusing on the specific use case of film review autocompletion.

### Models Implemented
- **Fine-tuned DistilGPT-2**: 82M parameters
- **Custom Transformer**: 8.8M parameters
  - 6 layers
  - 256 hidden dimensions
  - 4 attention heads

## Datasets
1. **IMDB Dataset**
   - 100,000 movie reviews (25,000 training, 25,000 testing, 50,000 unlabeled)
   - After preprocessing: 244,410 training sequences, 237,774 test sequences
   - Source: [Stanford IMDB Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)

2. **BookCorpus Subset**
   - 200 books from 7,185 available
   - 26,062 text chunks (avg. 97 words/chunk)
   - Source: [BookCorpus](https://huggingface.co/datasets/bookcorpus)

## Implementation Details
### DistilGPT-2 Fine-tuning
- Batch size: 4
- Gradient accumulation steps: 4
- Effective batch size: 16
- Training configurations: 5k, 10k, and 20k samples

### Custom Model
- Encoder-only architecture
- Word-level tokenization
- Trained on BookCorpus subset
- Fine-tuned on IMDB data

## Future Work
- Implement full dataset training
- Explore hybrid architectures
- Enhance generation quality
- Optimize for production deployment

## Requirements
- Python 3.8+
- PyTorch
- Transformers
- TensorFlow
- CUDA compatible GPU


## Acknowledgments
- HuggingFace for DistilGPT-2
- Stanford NLP for IMDB Dataset
- Google Colab for computational resources
