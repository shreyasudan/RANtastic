# Findings

### Summary Statistics
|                      |    naive |   finetuned |       rag |
|:---------------------|---------:|------------:|----------:|
| accuracy             |   0.6325 |      0.7225 |    0.703  |
| runtime  (in seconds)| 170.745  |   1248.2    |  2163.18  |

### Part A: Prompt Engineering and Inference Optimization

For the Basic Inference Engine task, I focused on optimizing inference performance while maintaining accuracy on the TeleQnA dataset. The key challenge was to efficiently serve the `Llama 3.2-3B instruct` model in a production-ready setup.

#### Model Serving Comparison

I implemented and compared two popular model serving frameworks:
1. `vLLM`: Emerged as the superior option for this task. Key advantages included:
  - Efficient batch processing (32 prompts per batch)
  - Optimized CUDA kernels for faster inference
  - Continuous batching that significantly reduced latency
  - Proper memory management with PagedAttention

2. `Ollama`: While easier to set up, it showed limitations:
  - Slower inference speeds despite multi-threading attempts
  - Less efficient memory management
  - Limited batch processing capabilities

#### Prompt Engineering

I experimented with several prompt configurations to improve accuracy:

- Implemented few-shot examples with telecommunications-specific content
- Added explicit instructions for output formatting (responding with only the option number)
- Included clear delimiters between question components
Used structured examples to demonstrate the desired answer format

The most effective prompt structure included:
```python
"""Answer the following multiple-choice question about telecommunications. Only respond with exactly one option (e.g., 'option 1', 'option 2', etc.) without any additional text.

Example 1:
Question: What is the capital of France?
option 1: Berlin
option 2: Madrid
option 3: Paris
option 4: Rome
option 5: Oslo
Answer: option 3

Example 2:
Question: What does HTTPS stand for?
option 1: Hypertext Transfer Protocol Secure
option 2: Hypertext Transfer Protocol Standard
option 3: Hypertext Transfer Protocol System
option 4: Hypertext Transfer Protocol Specification
option 5: Hypertext Transfer Protocol
Answer: option 1

Now, answer the question below in the same format:
Question: {question}
"""
```
#### Parallelization Strategies
To optimize runtime performance, I implemented:

- Batch processing of `32` questions at a time
- Asynchronous processing with proper concurrency management
- `ThreadPoolExecutor` for coordinating multiple inference requests
- Progress tracking with `tqdm` for monitoring batch processing
- `Regex`-based answer extraction for consistent response parsing

This parallelized approach with `vLLM` achieved an accuracy of `63.25%` with a runtime of just `170.7` seconds for the entire validation set.

---

### Part B: Fine-Tuning the LLM

#### Hyperparameter Optimization

I systematically explored various hyperparameters for LoRA fine-tuning:

**LoRA Configuration Experiments**
- Rank values: Tested `r=16` and `r=32`, with r=32 providing better performance
- Alpha settings: Set to 2x the rank value (`α=64` for `r=32`)
- Dropout rate: Tested `0.1`, `0.05`, and `0.01` as dropout rates. While `0.01` allowed for training time to be faster, it also led to overfitting to the training dataset, creating a considerable disparity between validation error and training error. I concluded that `0.05` was optimal as it balanced the tradeoff between runtime and performance of validation set.
- Target modules: Included query, key, value projection layers plus MLP components

#### Training Dynamics

- Step count analysis: Observed model convergence around 400-500 steps
- Overfitting detection: Monitored training vs. validation loss curves
- Epoch variation: Experimented with 3, 6, and 8 epochs, finding diminishing returns after 3
- Checkpoint evaluation: Implemented a function to evaluate all saved checkpoints to identify optimal model (`checkpoint-400`)

#### Error Analysis

**Category Error Rate**: Observed error rate by category to determine which categories are more likely to cause errors

| Category                | Error Rate | Notes  |
|:------------------------|------------|-------:|
| Standards overview      |   0.27     | Moderate difficulty with technical specs |
| Research publications   |   0.49     | Highest error rate - complex citations |
| Research overview       |   0.28     | Struggled with research methodology |
| Standards specifications|  0.37      | Technical details caused confusion |
| Lexicon                 |   0.13     | Lowest error rate - definitional content |

To address these category-specific errors, I modified the fine-tuning prompts to include category context:
```python
prompt = f"You're a telecommunications expert. Here's a multiple-choice question about telecommunications' {item['category'].lower() if item['category'] else ''}. Consider technical details carefully. Only respond with exactly one option without additional text."
```
This category-aware prompting significantly improved performance on the most challenging categories, particularly for Standards specifications and Research overview.

The fine-tuning process was executed using the `Unsloth` framework for efficiency, achieving a significant performance boost with an accuracy of `72.25%` on the validation set.

---

### Appendix: Retrieval-Augmented Generation (RAG) Implementation

#### Implementation Details and Architecture

I implemented a comprehensive RAG system to enhance the model's domain knowledge while maintaining inference efficiency:

1. **Document Processing Pipeline**:
  - Developed source-specific document processors for training data and 3GPP technical documents
  - Added enhanced metadata tagging to improve retrieval context awareness
  - Preserved document structure and relationships through intelligent chunking
2. **Embedding Strategy**:
  - Tested three OpenAI embedding models: `text-embedding-ada-002`, `text-embedding-3-small`, and `text-embedding-3-large`
  - Found `text-embedding-3-small` provided the optimal balance of performance and efficiency given the dataset size (`~3.9`MB)
  - Larger models like `text-embedding-3-large` did not provide sufficient accuracy gains to justify the increased computation time
3. **Retrieval Optimization**:
  - Implemented FAISS vector store for efficient similarity search
  - Extensive $k$-parameter testing revealed non-linear performance relationships
4. **Prompt Enhancement**: Integrated retrieved context into the prompt structure

#### Retrieval Parameter Experiments

The retriever's $k$ parameter (number of documents retrieved per query) had significant impact on model performance:

| k-Values| Accuracy| Runtime (s) | Notes   |
|:--------|---------|--------|-------------:|
| 3  |   0.6785     | 1873.44|Degraded performance vs. fine-tuned baseline|
| 4  |   0.703      | 2164.16|Optimal retrieval context size              |
| 7  |   0.703      | 2163.19|No additional gain with more documents      |
| 10 |  0.701       | 2560.29|Performance degradation with retrieval noise|


#### Chunking Strategy Analysis
A critical insight from my implementation was the importance of source-specific document chunking:

1. **Training Data**: Preserved QA pairs as atomic units to maintain the integrity of question-answer relationships:
```python
# Separate documents by source
# Keep training data intact - no chunking
training_docs = [doc for doc in documents if doc.metadata.get("source") == "training_data"]
# Don't chunk training data - keep QA pairs intact
```
2. **Technical Documents**: Implemented targeted chunking optimized for technical content:
```python
tech_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Optimized chunk size for technical content
    chunk_overlap=50,  # Sufficient overlap to maintain context
    separators=["\n\n", "\n", ". ", " ", ""]
)
```
This hybrid approach preserved the semantic structure of each document type, significantly improving retrieval relevance compared to a one-size-fits-all chunking strategy.

---

### Performance Comparision Analysis
Comparing the three approaches reveals important trade-offs:
1. **Basic Inference (Naive)**:
  - Fastest runtime (170.7s) but lowest accuracy (63.25%)
  - Excellent for quick responses but limited domain knowledge
2. **Fine-tuned Model**:
  - Best accuracy (72.25%) with moderate runtime (1248.2s)
  - Fixed knowledge embedded in model weights
3. **RAG Implementation**:
  - Good accuracy (70.3%) with longest runtime (2163.19s)
  - More flexible knowledge base that can be updated without retraining

The fine-tuned model outperformed RAG by approximately 2 percentage points while running nearly 2x faster. However, the RAG system offers key advantages in knowledge flexibility and extensibility that would be valuable in production scenarios where domain knowledge evolves rapidly.

---

### Conclusions and Production Recommendations

Based on these experiments, I recommend:

1. For static knowledge domains: Use the fine-tuned model for optimal performance-to-runtime ratio
2. For evolving domains: Use the RAG implementation with k=4 for balance of accuracy and efficiency
3. Consider a hybrid approach: Use the fine-tuned model as the base with RAG enhancement only for challenging questions or new domain knowledge areas

This tiered implementation strategy would maximize both accuracy and computational efficiency in a production environment.

---

### Future Considerations and Improvements

Based on the results and challenges encountered in this implementation, several promising avenues for improvement emerge:

1. **Hybrid Retrieval System**:
  - Implement a combination of dense and sparse retrievers (BM25 + embedding-based) to capture both keyword matching and semantic similarity
  - Add a re-ranking layer using cross-encoders to further refine the retrieval results before passing to the LLM
2. **Inference Optimization**:
  - Implement caching at multiple levels (embedding cache, retrieval cache, and response cache) to improve repeat question performance
  - Explore quantization methods like INT4/INT8 to further reduce latency without significant accuracy drops
3. **Adaptive Retrieval System**
  - Develop a confidence-based system that dynamically adjusts the retrieval parameters (k value) based on question difficulty
  - Implement query expansion techniques to improve retrieval for questions with limited context
4. **Knowledge Base Expansion**:
- Incorporate additional telecommunications standards beyond 3GPP
- Add specialized knowledge sources for categories where the model showed weaknesses

These improvements would provide promising directions for evolving this system into a more robust, accurate, and efficient production solution for telecommunications question answering.
