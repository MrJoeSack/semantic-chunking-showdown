import streamlit as st
import os
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(page_title="Chunking Showdown", layout="wide")

# Color scheme (mid-century modern / Mad Men aesthetic)
COLORS = {
    'primary': '#D2691E',      # Burnt orange (chocolate)
    'secondary': '#4A7C7E',    # Vintage teal
    'accent': '#DAA520',       # Goldenrod/mustard yellow
    'dark': '#3E2723',         # Dark chocolate brown
    'text': '#3E2723',         # Dark brown text
    'light_bg': '#F5E6D3',     # Warm cream/beige
    'border': '#8B7355',       # Medium brown border
    'fixed_bg': '#FFE4B5',     # Moccasin (warm peach)
    'semantic_bg': '#B0E0E6',  # Powder blue (retro)
    'warning': '#FF8C00',      # Dark orange
    'success': '#6B8E23'       # Olive green
}

# Custom CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    body {{
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: {COLORS['light_bg']};
    }}
    .stButton>button {{
        background-color: {COLORS['primary']};
        color: white;
        border: none;
        padding: 12px 28px;
        font-size: 16px;
        border-radius: 2px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 3px 3px 0px rgba(0,0,0,0.2);
    }}
    .stButton>button:hover {{
        background-color: {COLORS['secondary']};
        box-shadow: 4px 4px 0px rgba(0,0,0,0.25);
    }}
    .chunk-box {{
        border: 3px solid {COLORS['border']};
        padding: 18px;
        margin: 10px 0;
        border-radius: 0px;
        color: {COLORS['text']};
        font-size: 14px;
        line-height: 1.6;
        box-shadow: 5px 5px 0px rgba(0,0,0,0.1);
    }}
    .fixed-chunk {{
        background-color: {COLORS['fixed_bg']};
    }}
    .semantic-chunk {{
        background-color: {COLORS['semantic_bg']};
    }}
    .report-display {{
        background-color: {COLORS['light_bg']};
        color: {COLORS['text']};
        padding: 20px;
        border-radius: 0px;
        white-space: pre-wrap;
        border: 3px solid {COLORS['border']};
        font-family: 'Courier Prime', monospace;
        font-size: 14px;
        line-height: 1.6;
        box-shadow: 5px 5px 0px rgba(0,0,0,0.1);
    }}
    .callout {{
        background-color: #FFF4E6;
        border-left: 6px solid {COLORS['accent']};
        padding: 15px;
        margin: 10px 0;
        border-radius: 0px;
        color: {COLORS['text']};
        box-shadow: 3px 3px 0px rgba(0,0,0,0.1);
    }}
    .callout-bad {{
        background-color: #FFE6E6;
        border-left: 6px solid {COLORS['warning']};
    }}
    .callout-good {{
        background-color: #E6F7ED;
        border-left: 6px solid {COLORS['success']};
    }}
    h1, h2, h3 {{
        color: {COLORS['dark']};
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }}
    h1 {{
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 4px solid {COLORS['primary']};
        padding-bottom: 10px;
    }}
    .comparison-header {{
        font-weight: 600;
        color: {COLORS['dark']};
        margin-bottom: 10px;
        padding: 12px;
        background-color: {COLORS['light_bg']};
        border-radius: 0px;
        border: 2px solid {COLORS['border']};
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 12px;
    }}
</style>
""", unsafe_allow_html=True)

# Check for OpenAI API key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

@st.cache_data
def load_inspection_report():
    """
    Load optimized demo inspection report.
    This report is specifically designed to demonstrate chunking problems:
    - Clear semantic boundaries between sections
    - Sentences crafted to break awkwardly at 512-char fixed boundaries
    - Important context that gets fragmented with fixed chunking
    """
    return {
        'report_id': 999,
        'property_id': 42,
        'inspection_type': 'Pre-Sale',
        'full_report': """FOUNDATION: The foundation shows significant structural concerns that require immediate professional evaluation. Multiple vertical cracks extending 24-36 inches were observed in the southeast corner, accompanied by horizontal displacement of approximately half an inch. This pattern suggests active settlement issues rather than normal aging. The crawl space beneath shows standing water and inadequate drainage, with visible moisture damage to floor joists. These conditions create serious stability risks and potential for progressive structural failure if not addressed promptly.

ROOF: The asphalt shingle roof is approximately 18 years old and nearing the end of its serviceable life. Multiple shingles show curling, granule loss, and brittleness typical of age-related deterioration. More concerning are the damaged flashing around both chimneys and the improper valley installation that allows water penetration during heavy rainfall. The attic inspection revealed multiple water stains on decking and rafters, indicating chronic leakage. Immediate replacement is strongly recommended before the next rainy season to prevent interior water damage and potential mold growth.

ELECTRICAL: The 100-amp service panel is outdated and insufficient for modern electrical demands. Several circuits are double-tapped, creating fire hazards, and the panel shows signs of overheating with discolored breakers. The home contains a mix of copper and aluminum wiring, with several aluminum connections showing oxidation and loose terminations. Ground fault circuit interrupters are missing in bathrooms and kitchen as required by current codes. Three-prong outlets throughout the home lack proper grounding, creating shock hazards. A complete electrical system upgrade by a licensed electrician is necessary for both safety and code compliance.

PLUMBING: The galvanized steel supply pipes are severely corroded and restricting water flow throughout the home. Rust-colored water and diminished pressure indicate imminent pipe failure. The main sewer line shows signs of root intrusion based on slow drainage in multiple fixtures. The water heater is 14 years old, well past its expected 10-12 year lifespan, and showing rust around the base indicating internal tank corrosion. All fixtures require immediate attention to prevent catastrophic failures and water damage.

HVAC: The furnace is 22 years old with a cracked heat exchanger, creating an immediate carbon monoxide hazard. This is a life-safety issue requiring immediate shutdown and replacement. The air conditioning compressor is frozen and non-operational, likely due to refrigerant leaks and lack of maintenance. Ductwork shows extensive gaps and disconnections in the crawl space, resulting in massive energy loss. Beyond the equipment failures, the system is undersized for the square footage and has never been properly maintained, contributing to the current state of disrepair. Complete HVAC replacement is required before occupancy.

STRUCTURAL: Despite the foundation concerns, the upper structure shows good integrity. Interior walls are plumb and square with no signs of movement or distortion. Door frames operate properly without binding. The roof framing is adequately sized and properly constructed with no sagging or stress indicators. Floor systems are level and solid with appropriate joist spacing. Once foundation repairs are completed, the structural frame should provide many years of reliable service."""
    }

def fixed_chunking(text, chunk_size=512, overlap_pct=0.10):
    """
    TRULY fixed-size chunking with overlap.
    Breaks at exact character boundaries, even mid-word.
    This is the problem we're demonstrating.
    """
    overlap = int(chunk_size * overlap_pct)
    chunks = []

    # Calculate stride (how many chars to move forward each time)
    stride = chunk_size - overlap

    # Generate chunks at exact character positions
    position = 0
    while position < len(text):
        chunk = text[position:position + chunk_size]
        if chunk:  # Don't add empty chunks
            chunks.append(chunk)
        position += stride

    return chunks

def semantic_chunking(text):
    """
    LLM-based semantic chunking using GPT-4o-mini.

    This approach uses an LLM to intelligently identify semantic boundaries
    by actually understanding the content. The LLM reads the text and decides
    where major topic shifts occur, rather than relying on statistical measures.

    This demonstrates true semantic understanding - the model comprehends the
    content and makes intelligent splitting decisions based on meaning.
    """
    from openai import OpenAI
    import json

    client = OpenAI()

    # Ask the LLM to identify paragraph boundaries where topics shift
    prompt = f"""You are analyzing a document to identify semantic boundaries for chunking.

Read this text and identify where major topic shifts occur. Return the paragraph numbers (1-indexed) where you would split the text to create semantically coherent chunks.

Text:
{text}

Analyze the semantic structure and return ONLY a JSON array of paragraph numbers where splits should occur.
For example: [2, 4, 6] means split after paragraphs 2, 4, and 6.

Return ONLY the JSON array, no explanation."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # Deterministic
    )

    # Parse the response
    try:
        split_points = json.loads(response.choices[0].message.content)
    except:
        # Fallback: split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return paragraphs

    # Split text into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # Build chunks based on LLM's decisions
    chunks = []
    start = 0
    for split_point in sorted(split_points):
        if 0 < split_point <= len(paragraphs):
            chunk = '\n\n'.join(paragraphs[start:split_point])
            if chunk:
                chunks.append(chunk)
            start = split_point

    # Add final chunk
    final_chunk = '\n\n'.join(paragraphs[start:])
    if final_chunk:
        chunks.append(final_chunk)

    return chunks if chunks else [text]

def analyze_chunk_quality(fixed_chunks, semantic_chunks):
    """Analyze and identify specific issues with fixed vs semantic chunking"""
    issues = []

    # Check for mid-sentence breaks in fixed chunks
    for idx, chunk in enumerate(fixed_chunks):
        if not chunk.strip().endswith(('.', '!', '?', ':')):
            issues.append({
                'type': 'fixed',
                'chunk_index': idx,
                'issue': 'Breaks mid-sentence',
                'preview': chunk[-100:] if len(chunk) > 100 else chunk
            })

    # Check semantic chunks maintain topic boundaries
    for idx, chunk in enumerate(semantic_chunks):
        # Count section headers (all caps words at start)
        lines = chunk.split('\n')
        section_count = sum(1 for line in lines if line.strip() and line.strip().split(':')[0].isupper())
        if section_count > 1:
            issues.append({
                'type': 'semantic',
                'chunk_index': idx,
                'issue': f'Contains {section_count} complete sections',
                'preview': chunk[:100]
            })

    return issues

# Main app
st.title("Chunking Showdown: Fixed vs. Semantic")
st.markdown("### Demonstrating why chunking strategy matters for vector search")
st.markdown("_© 2025 Joe Sack Consulting LLC. All rights reserved._")

# Header links
st.markdown(f"""
<div style='text-align: center; margin: 10px 0;'>
    <a href='https://joesack.substack.com/p/fat-embeddings-weak-matches' target='_blank'>Read the Blog Post</a> |
    <a href='https://github.com/MrJoeSack/semantic-chunking-showdown' target='_blank'>View on GitHub</a>
</div>
""", unsafe_allow_html=True)

# Hero image
col1, col2, col3 = st.columns([3, 2, 3])
with col2:
    st.image("images/hero_chunking.png", use_container_width=True)

st.markdown("""
<div class='callout'>
<strong>About This Demo:</strong><br>
This is a <strong>standalone Python application</strong> built with Streamlit. It does not connect to SQL Server.
The fixed chunking is simulated in Python (matching SQL Server 2025's behavior), while semantic chunking
calls the OpenAI API live. The SQL code examples show you how to implement this in SQL Server 2025 when you're ready,
but everything you see here runs entirely in Python for easy exploration.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Why Embed? Primer
st.header("Why Do We Use Embeddings?")
st.markdown("_The foundation of modern semantic search_")

col1, col2, col3 = st.columns([3, 2, 3])
with col2:
    st.image("images/semantic_search.png", use_container_width=True)

st.markdown(f"""
<div style='background-color: {COLORS['light_bg']}; border: 2px solid {COLORS['border']}; padding: 20px; border-radius: 4px; margin-bottom: 20px;'>
<strong>The Challenge:</strong> Traditional keyword search can't understand meaning. Searching for "electrical issues" won't find
"power problems" or "circuit breaker failures" even though they're semantically related.
<br><br>
<strong>The Solution:</strong> Embedding models convert text into vectors (arrays of numbers) that capture semantic meaning.
Words with similar meanings have similar vectors. This lets us find relevant content based on meaning, not just matching keywords.
<br><br>
<strong>Example:</strong> The phrases "foundation crack" and "structural damage" will have similar embedding vectors because
they relate to the same concept, even though they share no words.
</div>
""", unsafe_allow_html=True)

# Why Chunk? Primer
st.header("Why Do We Chunk Text?")
st.markdown("_Breaking documents into searchable pieces_")

col1, col2, col3 = st.columns([3, 2, 3])
with col2:
    st.image("images/document_chunks.png", use_container_width=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style='background-color: #FFE6E6; border: 2px solid {COLORS['border']}; padding: 15px; border-radius: 4px; height: 200px;'>
    <strong>1. The Problem</strong><br><br>
    Documents are too large to process as a single unit. LLM context windows and embedding models have size limits.
    A 10-page report can't become a single embedding.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='background-color: #E6F7ED; border: 2px solid {COLORS['border']}; padding: 15px; border-radius: 4px; height: 200px;'>
    <strong>2. The Solution</strong><br><br>
    Break documents into smaller chunks. Each chunk becomes a separate embedding (a vector of numbers representing meaning).
    Store these vectors in a database.
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style='background-color: #E6F0FF; border: 2px solid {COLORS['border']}; padding: 15px; border-radius: 4px; height: 200px;'>
    <strong>3. The Search</strong><br><br>
    When users search, convert their query to a vector. Find chunks with similar vectors using distance calculations.
    Return the most relevant chunks.
    </div>
    """, unsafe_allow_html=True)

# Interactive Demo
st.header("See How Vector Search Works")
st.markdown("_Live demonstration using OpenAI's text-embedding-3-small model_")

col1, col2, col3 = st.columns([3, 2, 3])
with col2:
    st.image("images/vector_embeddings.png", use_container_width=True)

if OPENAI_KEY:
    from openai import OpenAI
    import numpy as np

    # Sample chunks for demonstration
    demo_chunks = [
        "The foundation shows significant structural concerns with multiple vertical cracks extending 24-36 inches in the southeast corner.",
        "The asphalt shingle roof is approximately 18 years old and showing signs of deterioration with curling shingles and granule loss.",
        "The electrical system has a 100-amp service panel that is outdated and shows signs of overheating with discolored breakers."
    ]

    st.markdown("**Sample Chunks from Inspection Report:**")
    for idx, chunk in enumerate(demo_chunks):
        st.markdown(f"**Chunk {idx+1}:** {chunk}")

    col1, col2 = st.columns([2, 1])

    with col1:
        user_query = st.text_input("Enter your search query:", placeholder="e.g., electrical problems")

    with col2:
        search_button = st.button("Search", type="primary")

    if search_button and user_query:
        with st.spinner("Generating embeddings and searching..."):
            client = OpenAI()

            # Generate embeddings for chunks
            chunk_embeddings = []
            for chunk in demo_chunks:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk
                )
                chunk_embeddings.append(response.data[0].embedding)

            # Generate embedding for query
            query_response = client.embeddings.create(
                model="text-embedding-3-small",
                input=user_query
            )
            query_embedding = query_response.data[0].embedding

            # Calculate cosine distance (like SQL Server 2025 VECTOR_DISTANCE)
            def cosine_distance(vec1, vec2):
                vec1 = np.array(vec1)
                vec2 = np.array(vec2)
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                # Convert similarity to distance: 0 = identical, higher = more different
                return 1 - similarity

            # Find most similar chunk (lowest distance)
            distances = [cosine_distance(query_embedding, chunk_emb) for chunk_emb in chunk_embeddings]
            best_match_idx = distances.index(min(distances))

        st.success("Search Complete!")

        st.markdown("**Embedding Details:**")
        st.markdown(f"- Each embedding has **1,536 dimensions** (numbers representing meaning)")
        st.markdown(f"- Query embedding preview: `[{', '.join([f'{query_embedding[i]:.4f}' for i in range(5)])}...]`")

        st.markdown("**Distance Scores (Cosine Distance - like SQL Server 2025):**")
        st.markdown("_Lower distance = more similar. 0.0 = identical, 1.0 = completely different_")
        for idx, dist in enumerate(distances):
            if idx == best_match_idx:
                st.markdown(f"""
                <div class='callout callout-good'>
                <strong>Best Match - Chunk {idx+1}</strong> (Distance: {dist:.4f})<br>
                {demo_chunks[idx]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"**Chunk {idx+1}** (Distance: {dist:.4f}): {demo_chunks[idx]}")

        st.info("**Key Insight:** SQL Server 2025's VECTOR_DISTANCE returns the chunk with the **lowest distance score**. This matches how ORDER BY distance ASC works in SQL Server vector search queries.")

else:
    st.warning("OpenAI API key required to run the interactive search demo. Set OPENAI_API_KEY in your .env file.")

st.markdown("---")

# Load report
report = load_inspection_report()

# Display original document
st.header("Now Let's Compare Chunking Strategies")
st.markdown("_How you chunk affects search quality_")

st.subheader("Demo Inspection Report")
st.info(f"**Report ID:** {report['report_id']} (demo) | **Property ID:** {report['property_id']} | **Type:** {report['inspection_type']}")
st.markdown("_This report is optimized to clearly demonstrate fixed vs semantic chunking differences._")

with st.expander("View Full Report", expanded=False):
    st.markdown(f"<div class='report-display'>{report['full_report']}</div>", unsafe_allow_html=True)

st.markdown("---")

# Show code implementations FIRST
st.header("💻 Implementation Code")
st.markdown("_Reference examples: SQL Server 2025 syntax (left) and Python with OpenAI (right). This demo simulates both approaches in Python._")

code_col1, code_col2 = st.columns(2)

with code_col1:
    st.subheader("SQL Server 2025 (Fixed)")
    st.markdown("_SQL Server 2025 provides AI_GENERATE_CHUNKS for fixed-size chunking with overlap. (Reference code - simulated in Python for this demo)_")
    st.code("""
-- Fixed chunking in SQL Server 2025
-- Uses built-in AI_GENERATE_CHUNKS function (Preview)

-- Enable preview features (required)
ALTER DATABASE SCOPED CONFIGURATION
SET PREVIEW_FEATURES = ON;

-- Chunk text from a table
SELECT
    r.report_id,
    c.chunk,
    c.chunk_order,
    c.chunk_offset,
    c.chunk_length
FROM SemanticInspectDB.dbo.inspection_reports r
CROSS APPLY AI_GENERATE_CHUNKS(
    source = r.full_report,
    chunk_type = FIXED,
    chunk_size = 512,
    overlap = 10  -- 10% overlap
) AS c
WHERE r.report_id = 999
ORDER BY c.chunk_order;

-- Note: Currently only FIXED chunking is supported
-- Future versions may add semantic chunking options
    """, language="sql")

with code_col2:
    st.subheader("Python (Semantic)")
    st.markdown("_LLM-based semantic chunking uses language understanding to identify topic boundaries. (Live execution via OpenAI API)_")
    st.code("""
from openai import OpenAI
import json

client = OpenAI()

# Ask the LLM to identify semantic boundaries
prompt = f\"\"\"Analyze this text and identify where major
topic shifts occur. Return paragraph numbers where you
would split the text.

Text: {text}

Return ONLY a JSON array like: [1, 2, 3, 4, 5]\"\"\"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0  # Deterministic
)

# Parse LLM's decisions
split_points = json.loads(response.choices[0].message.content)

# Split text at LLM-identified boundaries
paragraphs = text.split('\\n\\n')
chunks = []
start = 0
for split_point in split_points:
    chunk = '\\n\\n'.join(paragraphs[start:split_point])
    chunks.append(chunk)
    start = split_point

# Result: True semantic understanding based on content
    """, language="python")

st.markdown("---")

# Check for OpenAI API key
if not OPENAI_KEY:
    st.error("**OpenAI API key required.** Set the `OPENAI_API_KEY` environment variable to run this demo.")
    st.code("# Create a .env file with:\nOPENAI_API_KEY=your-key-here", language="bash")
    st.stop()

# Generate both chunk types
if st.button("🚀 Run Chunking Comparison", type="primary"):
    # Get the source text
    source_text = report['full_report']

    # Debug: Show source text info
    st.info(f"📝 Processing report (ID {report['report_id']}): {len(source_text)} characters, {len(source_text.split())} words")

    with st.spinner("Generating fixed chunks..."):
        start_time = time.time()
        fixed_chunks = fixed_chunking(source_text)
        fixed_time = time.time() - start_time

    with st.spinner("Generating semantic chunks (analyzing sentence embeddings)..."):
        start_time = time.time()
        semantic_chunks = semantic_chunking(source_text)
        semantic_time = time.time() - start_time

    with st.spinner("Analyzing chunk quality..."):
        issues = analyze_chunk_quality(fixed_chunks, semantic_chunks)

    # Store in session state
    st.session_state.fixed_chunks = fixed_chunks
    st.session_state.semantic_chunks = semantic_chunks
    st.session_state.issues = issues
    st.session_state.fixed_time = fixed_time
    st.session_state.semantic_time = semantic_time
    st.session_state.source_text = source_text  # Store for verification

    st.success(f"Chunking complete! Fixed: {fixed_time:.2f}s | Semantic: {semantic_time:.2f}s")

# Display side-by-side comparison
if 'fixed_chunks' in st.session_state and 'semantic_chunks' in st.session_state:
    fixed_chunks = st.session_state.fixed_chunks
    semantic_chunks = st.session_state.semantic_chunks
    issues = st.session_state.issues

    st.header("📊 Side-by-Side Comparison")

    # Summary metrics
    st.markdown("_These metrics show the differences between the two approaches._")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fixed Chunks", len(fixed_chunks), help="512 chars, 10% overlap")
    with col2:
        st.metric("Semantic Chunks", len(semantic_chunks), help="Variable size, topic-based")
    with col3:
        st.metric("Issues Found", len([i for i in issues if i['type'] == 'fixed']),
                 help="Mid-sentence breaks in fixed chunks")
    with col4:
        fixed_time = st.session_state.get('fixed_time', 0)
        semantic_time = st.session_state.get('semantic_time', 0)
        st.metric("Time (Fixed/Semantic)", f"{fixed_time:.2f}s / {semantic_time:.2f}s",
                 help="Processing time for each approach")

    st.markdown("---")

    # Show specific examples
    st.subheader("🔍 Chunk-by-Chunk Analysis")
    st.markdown("_Let's walk through the chunks. Left side breaks arbitrarily. Right side breaks naturally._")

    # Display chunks side by side
    max_chunks = max(len(fixed_chunks), len(semantic_chunks))

    # Calculate additional metrics
    # Visual comparison
    col1, col2, col3 = st.columns([3, 2, 3])
    with col2:
        st.image("images/comparison_visual.png", use_container_width=True)

    fixed_avg_size = sum(len(c) for c in fixed_chunks) / len(fixed_chunks)
    semantic_avg_size = sum(len(c) for c in semantic_chunks) / len(semantic_chunks)

    # Add metric row for chunk sizes
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fixed Avg Size", f"{fixed_avg_size:.0f} chars")
    with col2:
        st.metric("Semantic Avg Size", f"{semantic_avg_size:.0f} chars")
    with col3:
        st.metric("Fixed Min/Max", f"{min(len(c) for c in fixed_chunks)}-{max(len(c) for c in fixed_chunks)}")
    with col4:
        st.metric("Semantic Min/Max", f"{min(len(c) for c in semantic_chunks)}-{max(len(c) for c in semantic_chunks)}")

    st.markdown("---")

    for i in range(max_chunks):  # Show ALL chunks
        st.markdown(f"### Chunk Set {i + 1}")

        col_fixed, col_semantic = st.columns(2)

        with col_fixed:
            st.markdown(f"<div class='comparison-header'>Fixed Chunking</div>", unsafe_allow_html=True)
            if i < len(fixed_chunks):
                chunk = fixed_chunks[i]
                st.markdown(f"""
                <div class='chunk-box fixed-chunk'>
                <strong>Chunk {i + 1}</strong> ({len(chunk)} chars)<br><br>
                {chunk}
                </div>
                """, unsafe_allow_html=True)

                # Check for issues
                chunk_issues = [issue for issue in issues if issue['type'] == 'fixed' and issue['chunk_index'] == i]
                if chunk_issues:
                    for issue in chunk_issues:
                        st.markdown(f"""
                        <div class='callout callout-bad'>
                        <strong>Problem:</strong> {issue['issue']}<br>
                        <em>This breaks context mid-thought, hurting embedding quality.</em>
                        </div>
                        """, unsafe_allow_html=True)

        with col_semantic:
            st.markdown(f"<div class='comparison-header'>Semantic Chunking</div>", unsafe_allow_html=True)
            if i < len(semantic_chunks):
                chunk = semantic_chunks[i]
                st.markdown(f"""
                <div class='chunk-box semantic-chunk'>
                <strong>Chunk {i + 1}</strong> ({len(chunk)} chars)<br><br>
                {chunk}
                </div>
                """, unsafe_allow_html=True)

                # Check if this chunk preserves boundaries
                ends_complete = chunk.strip() and chunk.strip()[-1] in '.!?:'
                if ends_complete:
                    st.markdown(f"""
                    <div class='callout callout-good'>
                    <strong>Observation:</strong> Ends at sentence boundary<br>
                    <em>Context-aware splitting based on similarity.</em>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

    # Now show the analysis after seeing all chunks
    st.header("🔍 What Did We Just See?")
    st.markdown("_Here's an example showing how fixed chunking breaks at character boundaries._")

    fixed_issues = [i for i in issues if i['type'] == 'fixed']
    if fixed_issues:
        example = fixed_issues[0]  # First issue as example
        st.markdown(f"""
        <div class='callout callout-bad'>
        <strong>Example: Fixed Chunk #{example['chunk_index'] + 1}</strong><br>
        {example['issue']} - Character-based splitting doesn't consider sentence boundaries.<br>
        <em>Preview: "...{example['preview'][-80:]}"</em>
        </div>
        """, unsafe_allow_html=True)

# Final Summary Section
if 'fixed_chunks' in st.session_state and 'semantic_chunks' in st.session_state:
    st.markdown("---")
    st.header("📋 Key Takeaways")
    st.markdown("_Here are the tradeoffs. The right choice depends on your specific requirements._")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style='background-color: {COLORS['fixed_bg']}; border: 2px solid {COLORS['border']}; padding: 15px; border-radius: 4px;'>
        <strong>Fixed Chunking</strong><br><br>
        <strong>Best For:</strong><br>
        • Large-scale processing (millions of documents)<br>
        • Deterministic results required<br>
        • Budget constraints (no API costs)<br>
        • Real-time requirements (< 1 second)<br><br>
        <strong>Limitations:</strong><br>
        • Breaks at character boundaries (may split mid-sentence)<br>
        • Ignores semantic meaning<br>
        • Can fragment context and reduce search accuracy
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background-color: {COLORS['semantic_bg']}; border: 2px solid {COLORS['border']}; padding: 15px; border-radius: 4px;'>
        <strong>Semantic Chunking</strong><br><br>
        <strong>Best For:</strong><br>
        • Content with clear topic boundaries<br>
        • Quality over speed priority<br>
        • Smaller datasets (thousands of documents)<br>
        • Complex documents where context matters<br><br>
        <strong>Limitations:</strong><br>
        • Requires API calls (~$0.0001 per document)<br>
        • Slower processing (2-5 seconds per document)<br>
        • Variable chunk sizes (harder to predict storage)
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='callout'>
    <strong>Bottom Line:</strong> Fixed chunking prioritizes speed and simplicity. Semantic chunking prioritizes search accuracy and context preservation.
    Choose based on your scale, budget, and quality requirements.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: {COLORS['dark']}; padding: 20px 0;'>
    <strong>Built by Joe Sack Consulting LLC</strong><br>
    <a href='https://joesack.substack.com/p/fat-embeddings-weak-matches' target='_blank'>Read the Blog Post</a> |
    <a href='https://github.com/MrJoeSack/semantic-chunking-showdown' target='_blank'>View on GitHub</a> |
    <a href='https://joesack.substack.com' target='_blank'>Joe's Substack</a>
</div>
""", unsafe_allow_html=True)
