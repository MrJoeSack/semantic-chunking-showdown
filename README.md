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

## Live Demo Results

**Inspection Report Chunking:**

| Method | Chunks | Approach | Time |
|--------|--------|----------|------|
| Fixed | 6 chunks @ 512 chars | Character boundaries | ~0.1s |
| Semantic | 6 chunks @ 482-655 chars | Topic boundaries (LLM) | ~5s |

**Fixed chunking** breaks at exact character positions, potentially splitting words or sentences.
**Semantic chunking** creates variable-sized chunks that respect complete sections (FOUNDATION, ROOF, ELECTRICAL, PLUMBING, HVAC, STRUCTURAL).

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

## Technical Details

### Fixed Chunking Implementation

```python
def fixed_chunking(text, chunk_size=512, overlap_pct=0.10):
    """Splits at exact character boundaries, even mid-word"""
    overlap = int(chunk_size * overlap_pct)
    stride = chunk_size - overlap

    chunks = []
    position = 0
    while position < len(text):
        chunk = text[position:position + chunk_size]
        if chunk:
            chunks.append(chunk)
        position += stride

    return chunks
```

### Semantic Chunking Implementation

```python
def semantic_chunking(text):
    """Uses GPT-4o-mini to identify semantic boundaries"""
    from openai import OpenAI
    import json

    client = OpenAI()

    prompt = f"""Analyze this text and identify where major
    topic shifts occur. Return paragraph numbers where you
    would split the text.

    Text: {text}

    Return ONLY a JSON array like: [1, 2, 3, 4, 5]"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # Deterministic
    )

    split_points = json.loads(response.choices[0].message.content)

    # Split text at LLM-identified boundaries
    paragraphs = text.split('\n\n')
    chunks = []
    start = 0
    for split_point in split_points:
        chunk = '\n\n'.join(paragraphs[start:split_point])
        chunks.append(chunk)
        start = split_point

    return chunks
```

## SQL Server 2025 Reference

The demo shows how SQL Server 2025's `AI_GENERATE_CHUNKS` function works:

```sql
-- Enable preview features
ALTER DATABASE SCOPED CONFIGURATION
SET PREVIEW_FEATURES = ON;

-- Fixed chunking with AI_GENERATE_CHUNKS
SELECT
    c.chunk,
    c.chunk_order,
    c.chunk_offset,
    c.chunk_length
FROM inspection_reports r
CROSS APPLY AI_GENERATE_CHUNKS(
    source = r.full_report,
    chunk_type = FIXED,
    chunk_size = 512,
    overlap = 10  -- 10% overlap
) AS c
WHERE r.report_id = 999
ORDER BY c.chunk_order;
```

**Note:** SQL Server 2025 currently only supports FIXED chunking. Semantic chunking requires external processing (like the Python/LLM approach shown in this demo).

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
