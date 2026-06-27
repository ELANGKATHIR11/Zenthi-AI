# Router Classification Accuracy Report

This automated test report details the semantic routing accuracy of **Zenthi-AI OS** across **500 evaluation test queries** (100 queries per intent category).

## Benchmark Performance Summary

- **Total Queries Evaluated**: 500
- **Evaluation Time**: 325.77 seconds
- **Average Latency**: 651.54 ms
- **Overall Routing Accuracy**: 72.60%

## Category Performance Breakdown

| Intent Category | Correct / Total | Category Accuracy | Target Expert Model |
| :--- | :--- | :--- | :--- |
| **CODE** | 100 / 100 | 100.00% | `qwen2.5-coder:3b` |
| **RAG** | 43 / 100 | 43.00% | `qwen2.5:1.5b-instruct` |
| **SEARCH** | 99 / 100 | 99.00% | `qwen2.5:1.5b-instruct` |
| **VISION** | 100 / 100 | 100.00% | `riven/smolvlm:latest` |
| **KNOWLEDGE** | 21 / 100 | 21.00% | `qwen2.5:1.5b-instruct` |

*Report generated at run_evaluation execution completion.*
