# LLM Text Compressor

A novel text compression system that leverages Large Language Models (specifically GPT-2) to achieve high compression ratios by predicting token sequences and storing only prediction ranks instead of raw tokens.

## Overview

This project implements an innovative approach to text compression that combines the predictive power of modern language models with traditional compression techniques. Instead of storing the actual tokens, the system stores the rank of each token within the language model's probability-ordered predictions, which are then further compressed using arithmetic coding.

## Key Features

- **High Compression Ratio**: Achieves ~74% compression on test data (Alice in Wonderland: 41,790 bytes → 10,620 bytes)
- **Lossless Compression**: Perfect reconstruction of original text
- **LLM-Powered**: Uses GPT-2's predictive capabilities for intelligent compression
- **Sliding Window Approach**: Maintains context with configurable memory window size
- **Arithmetic Coding**: Secondary compression layer for optimal storage efficiency

## How It Works

### Compression Process

1. **Tokenization**: Input text is tokenized using GPT-2's tokenizer
2. **Sliding Window**: A memory window of size M (default: 16) slides through the token sequence
3. **Prediction**: For each position, GPT-2 predicts the probability distribution of the next token
4. **Rank Calculation**: Instead of storing the actual token, the system stores its rank in the sorted probability distribution
5. **Arithmetic Coding**: The sequence of ranks is further compressed using arithmetic coding
6. **Storage**: The compressed data is stored in a binary file

### Decompression Process

1. **Decode**: Arithmetic coding is reversed to recover the rank sequence
2. **Reconstruction**: For each rank, GPT-2 generates predictions and selects the token at that rank
3. **Sliding Window**: The context window is updated with each predicted token
4. **Detokenization**: The token sequence is converted back to readable text

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for performance)
- ~2GB free disk space for GPT-2 model

### Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

The main dependencies include:
- `torch` - PyTorch framework
- `transformers` - Hugging Face Transformers for GPT-2
- `numpy` - Numerical computations
- Standard library modules for file I/O and data structures

## Usage

### Basic Usage

The system is designed to work out-of-the-box with the provided sample text:

```bash
python main.py
```

This will:
1. Load the GPT-2 model and tokenizer
2. Read `alice_in_wonderland.txt`
3. Compress it to `compressed.bin`
4. Decompress it to `decompressed.txt`

### Custom Text Compression

To compress your own text file:

1. Replace `alice_in_wonderland.txt` with your text file, or
2. Modify the filename in `main.py` (line 17):

```python
with open("your_text_file.txt", "r") as file:
    text = file.read()
```

### Configuration Options

#### Memory Window Size (M)

Adjust the context window size by modifying the `M` parameter in `main.py`:

```python
M = 16  # Default value, can be adjusted for different compression/quality tradeoffs
```

- **Larger M**: Better context, potentially better compression, but slower processing
- **Smaller M**: Faster processing, but may reduce compression efficiency

#### Arithmetic Coding Precision

Modify the precision in `arithmetic_coding.py`:

```python
class ArithmeticCoder:
    def __init__(self, precision=32):  # Adjust precision as needed
```

## Project Structure

```
LLM-Text-Compressor/
│
├── main.py                 # Main entry point and orchestration
├── compress.py             # Core compression logic
├── decompress.py           # Core decompression logic
├── arithmetic_coding.py    # Arithmetic coding implementation
├── requirements.txt        # Python dependencies
├── alice_in_wonderland.txt # Sample input text
├── compressed.bin          # Compressed output (generated)
├── decompressed.txt        # Decompressed output (generated)
└── README.md              # This documentation
```

## Technical Details

### Algorithm Components

#### 1. LLM Rank Compression (`compress.py`)

```python
def compress(input_ids, model, M=4) -> List[int]:
```

- Uses sliding window approach with memory size M
- For each token position, generates GPT-2 predictions
- Computes rank of actual token in sorted prediction probabilities
- Returns list of ranks instead of original tokens

#### 2. LLM Rank Decompression (`decompress.py`)

```python
def decompress(ranks, input_ids, tokenizer, model, M=4) -> str:
```

- Reconstructs text by using ranks to select tokens from GPT-2 predictions
- Maintains sliding window context during reconstruction
- Returns fully reconstructed text string

#### 3. Arithmetic Coding (`arithmetic_coding.py`)

Implements standard arithmetic coding with:
- **Encoding**: Converts rank sequence to single compressed integer
- **Decoding**: Recovers original rank sequence from compressed data
- **File I/O**: Handles binary storage with frequency tables

### Performance Metrics

Based on the included Alice in Wonderland sample:

| Metric | Value |
|--------|-------|
| Original Size | 41,790 bytes |
| Compressed Size | 10,620 bytes |
| Compression Ratio | ~74.6% |
| Space Savings | ~25.4% of original |

### Memory Requirements

- **GPU Memory**: ~2GB for GPT-2 model
- **System RAM**: ~1GB for processing
- **Disk Space**: Original text size + ~25% for compressed output

### Processing Time

Processing time scales with:
- Text length (linear)
- Memory window size M (linear)
- GPU performance (significant impact)

## Examples

### Compression Example

```python
from compress import compress
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Compress text
text = "Your text here..."
input_ids = tokenizer.encode(text, return_tensors="pt")
ranks = compress(input_ids, model, M=16)
```

### Decompression Example

```python
from decompress import decompress

# Decompress ranks back to text
reconstructed_text = decompress(ranks, input_ids[0][:M], tokenizer, model, M)
```

## Limitations and Considerations

### Current Limitations

1. **GPU Dependency**: Requires CUDA-compatible GPU for practical performance
2. **Model Size**: GPT-2 model requires significant disk space and memory
3. **Processing Speed**: Compression/decompression is slower than traditional algorithms
4. **Text Domain**: Performance may vary significantly across different text types

### Best Use Cases

- **Academic Research**: Novel compression algorithm research
- **Long-form Text**: Books, articles, documents with rich linguistic structure
- **Educational Purposes**: Understanding LLM applications in compression

### Not Recommended For

- **Real-time Applications**: Due to processing overhead
- **Binary Data**: Designed specifically for natural language text
- **Short Text Snippets**: Overhead may exceed benefits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/SatvikG7/LLM-Text-Compressor.git
cd LLM-Text-Compressor
pip install -r requirements.txt
```

## License

This project is open source. Please refer to the repository license for specific terms.

## Research and References

This implementation is based on the concept of using language model predictions for text compression. The approach demonstrates how modern NLP models can be applied to traditional computer science problems like data compression.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use smaller memory window (M)
2. **Model Download Issues**: Ensure stable internet connection for initial GPT-2 download
3. **Performance Issues**: Verify CUDA installation and GPU availability

### Getting Help

- Check that all dependencies are correctly installed
- Verify GPU drivers and CUDA installation
- Ensure sufficient disk space for model and output files

---

*For questions, issues, or contributions, please visit the project repository.*