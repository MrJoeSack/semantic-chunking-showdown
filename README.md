# Chunking Showdown: Fixed vs. Semantic

An interactive demonstration comparing fixed-size chunking (SQL Server 2025) with LLM-based semantic chunking (GPT-4o-mini) for vector search applications.

## Blog Post

Read the accompanying article: **[Fat Embeddings, Weak Matches](https://joesack.substack.com/p/fat-embeddings-weak-matches)**

## What This Demonstrates

This demo shows a fundamental tradeoff in vector search: **speed and simplicity vs. semantic accuracy**.

- **Fixed Chunking (SQL Server 2025)**: Splits text at exact character boundaries (512 chars). Fast, deterministic, but may break mid-sentence.
- **Semantic Chunking (LLM-based)**: Uses GPT-4o-mini to intelligently identify topic boundaries. Slower, costs money, but preserves complete thoughts.

## Key Innovation: LLM-Based Semantic Chunking

Instead of using statistical methods (embedding similarity, cosine distance), this demo uses **GPT-4o-mini to actually read the document** and identify where topics shift. The LLM comprehends content semantically and makes intelligent splitting decisions.

**Why This Works Better:**
- Statistical methods struggled to find the right threshold for section-level splits
- All inspection report sections had similar vocabulary, making cosine similarity ineffective
- LLM approach uses actual language understanding, not just vector math
- Produces clean, human-like topic segmentation

## Prerequisites

- Python 3.9+
- OpenAI API key with access to `gpt-4o-mini`
- (Optional) SQL Server 2025 for exploring AI_GENERATE_CHUNKS

## Installation

```bash
git clone https://github.com/MrJoeSack/semantic-chunking-showdown.git
cd semantic-chunking-showdown
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

**Note:** Your OpenAI project must have access to `gpt-4o-mini`. Enable this in your OpenAI dashboard under Project → Limits → Model usage.

## Usage

```bash
streamlit run app.py
```

The app will:
1. Display a demo inspection report (optimized for clear chunking comparison)
2. Show implementation code for both approaches
3. Generate fixed and semantic chunks when you click "Run Chunking Comparison"
4. Display side-by-side comparison with metrics and analysis
5. Highlight specific issues (mid-sentence breaks in fixed chunking)

## Architecture

**Frontend:**
- Streamlit (interactive web interface)

**Fixed Chunking (Conceptual):**
- SQL Server 2025 `AI_GENERATE_CHUNKS` function
- Exact character-based splitting with overlap
- Code example shown (not executed in demo)

**Semantic Chunking (Live):**
- OpenAI GPT-4o-mini API
- LLM reads text and identifies topic boundaries
- Returns paragraph numbers for splitting
- Executed live in the demo

## Key Takeaways

**When to Use Fixed Chunking:**
- Large-scale batch processing (millions of documents)
- Need for deterministic, reproducible results
- Budget constraints (no API costs)
- Real-time requirements (< 1 second)
- Simple, predictable content

**When to Use Semantic Chunking:**
- Content with clear topic boundaries
- Quality matters more than speed
- Smaller datasets (thousands, not millions)
- Willing to pay API costs (~$0.0001 per document)
- Complex documents where context is critical

## Project Structure

```
semantic-chunking-showdown/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                           # API keys (gitignored)
├── .streamlit/
│   └── config.toml                # Streamlit theme configuration
├── README.md                      # This file
├── LLM_SEMANTIC_CHUNKING_COMPLETE.md  # Technical implementation notes
├── FLOW_RESTRUCTURE.md            # UI flow documentation
└── test_*.py                      # Testing scripts
```

## Contributing

This is a demonstration project for educational purposes. Feel free to fork and adapt for your own use cases.

## Resources

- **Blog Post:** [Fat Embeddings, Weak Matches](https://joesack.substack.com/p/fat-embeddings-weak-matches)
- **SQL Server 2025 Docs:** [AI_GENERATE_CHUNKS](https://learn.microsoft.com/en-us/sql/t-sql/functions/ai-generate-chunks-transact-sql)
- **OpenAI API:** [Chat Completions](https://platform.openai.com/docs/guides/chat-completions)

## Author

**Joe Sack**
joe@sackhq.com
[joesack.substack.com](https://joesack.substack.com)
[@MrJoeSack](https://twitter.com/MrJoeSack)

## License

MIT License - See LICENSE file for details

---

*Built for the "Fat Embeddings, Weak Matches" blog post demonstrating real-world chunking tradeoffs for SQL Server 2025 vector search.*
