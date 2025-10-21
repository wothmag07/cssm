# Amazon Product Assistant - Customer Support Bot

A RAG-based customer support bot with Amazon-style UI that helps customers find the best electronics products based on real customer reviews.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Using uv Package Manager](#using-uv-package-manager)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Frontend Setup](#frontend-setup)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Usage Examples](#usage-examples)
- [Scaling Up](#scaling-up)
- [Project Status](#project-status)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Support](#support)

---

## Overview

This project combines:
- **FastAPI Backend** - RAG pipeline with vector search
- **Next.js Frontend** - Modern React/TypeScript UI
- **AstraDB** - Vector database for product reviews
- **LangChain** - RAG orchestration
- **Google Gemini** - LLM for intelligent responses

---

## Architecture

### High-Level Overview

```text
[ Browser ]
  ▲  │ fetch /api/retrieve or /api/retrieve/stream
  │  ▼
[ Next.js Route Handler / Server Action (TypeScript) ]  ──HTTP──>  [ FastAPI Service (Python) ]
    ▲                                                                         │
    │                                                                         │  /retrieve
    │                                                                         ▼
    │                                 [ [ embed ] → [ hybrid search @ AstraDB ] → [ MMR ] → [ rerank ] ]
    │                                                                   │
    │                                                              results (JSON/SSE)
    │                                                                   │
    └────────────────────────────── JSON / SSE response ────────────────┘

```
*Figure 1: Application Flow*

![RAG Pipeline Diagram](images/simple-local-rag-workflow-flowchart.png)
*Figure 2: RAG Pipeline Conceptual Flow*

---

## Dataset

### Amazon Product Reviews Dataset (2023)

This project uses the **Amazon Product Reviews Dataset** from 2023 by **McAuley Lab**, specifically the **Electronics** category.

**Dataset Details:**
- **Official Source**: [Amazon Reviews'23](https://amazon-reviews-2023.github.io/)
- **Research Lab**: McAuley Lab, UC San Diego
- **Category**: Electronics
- **Total Reviews in Dataset**: 571.54M (across all categories)
- **Electronics Reviews**: ~20.8M reviews
- **Electronics Users**: ~11.6M users
- **Electronics Items**: ~1.3M items
- **Timespan**: May 1996 - September 2023
- **Citation**: [Bridging Language and Items for Retrieval and Recommendation](https://arxiv.org/abs/2403.03952) (arXiv:2403.03952, 2024)

**Dataset Structure:**

The dataset consists of two files:
1. **Reviews File** (`Electronics.jsonl`): Customer reviews with ratings
2. **Metadata File** (`meta_Electronics.jsonl`): Product information

**Review Schema:**
```json
{
  "asin": "B000001234",
  "user_id": "A1B2C3D4E5F6",
  "text": "Great laptop for students...",
  "title": "Excellent value",
  "rating": 5,
  "verified_purchase": true,
  "helpful_vote": 10,
  "timestamp": 1234567890
}
```

**Metadata Schema:**
```json
{
  "parent_asin": "B000001234",
  "asin": "B000001234",
  "title": "Acer Aspire 5 Laptop",
  "description": ["15.6 inch", "8GB RAM", "256GB SSD"],
  "main_category": "Computers",
  "average_rating": 4.5,
  "rating_number": 1250,
  "price": "$499.99",
  "store": "Acer Store"
}
```

### Data Processing Pipeline

**Step 1: Merge Reviews with Metadata**

The `data/data.py` script merges review data with product metadata:

```bash
python data/data.py \
  --reviews data/Electronics.jsonl \
  --metadata data/meta_Electronics.jsonl \
  --out_json data/merged_electronics_data.json \
  --out_jsonl data/merged_electronics_data.jsonl
```

**What it does:**
- Loads product metadata and indexes by ASIN
- Merges each review with its corresponding product metadata
- Creates enriched records with product details
- Handles malformed JSON gracefully
- Outputs both JSON (array) and JSONL (line-delimited) formats
- Supports sampling with `--limit` flag

**Merged Record Schema:**
```json
{
  "product_id": "B000001234",
  "product_name": "Acer Aspire 5 Laptop",
  "product_description": "15.6 inch 8GB RAM 256GB SSD",
  "user_id": "A1B2C3D4E5F6",
  "text": "Great laptop for students...",
  "title": "Excellent value",
  "rating": 5,
  "avg_rating": 4.5,
  "rating_count": 1250,
  "category": "Computers",
  "store": "Acer Store",
  "price": "$499.99",
  "verified_purchase": true,
  "helpful_vote": 10,
  "timestamp": 1234567890
}
```

**Step 2: Data Ingestion into Vector Database**

The merged data is then ingested into AstraDB:

```bash
python data-ingestion/data_ingestion.py
```

**What it does:**
- Loads merged JSONL file
- Transforms reviews into Document objects
- Generates embeddings using Google text-embedding-004
- Stores vectors + metadata in AstraDB
- Batch processing with retry logic
- Verifies ingestion with test query

**Data Statistics:**
- **Total Records**: 5,000,000
- **Laptop Records**: 397,334 (7.9%)
- **Average Rating**: 4.2/5
- **Verified Purchases**: ~85%
- **Date Range**: 2023

**Dataset License:**
- Publicly available from McAuley Lab
- For research and educational purposes
- Collected by McAuley Lab, UC San Diego

**Download the Dataset:**
```bash
# Option 1: Download from official source
# Electronics Reviews
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz

# Electronics Metadata
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz

# Option 2: Use the pre-processed files in data/ directory
# The merged files are already included in the repository
```

**Official Resources:**
- [Official Dataset Website](https://amazon-reviews-2023.github.io/)
- [Hugging Face Datasets](https://huggingface.co/datasets/amazon-reviews-2023)
- [GitHub Repository](https://github.com/amazon-reviews-2023)
- [Research Paper](https://arxiv.org/abs/2403.03952)

---

## Prerequisites

- **Python 3.10+** installed
- **Node.js 18+** installed
- **AstraDB account** (free tier at [astra.datastax.com](https://astra.datastax.com))
- **API Keys** for at least one LLM provider:
  - Google Gemini (recommended)
  - Groq (fast and free)
  - OpenAI
  - Anthropic
- **Optional:** `uv` package manager for faster setup (`pip install uv`)

---

## Quick Start

### 1. Setup Backend

**Using `uv` (Recommended - 10-100x faster):**
```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment and install dependencies
uv venv && uv pip install -r requirements.txt

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Copy environment template and add your API keys
copy env.template .env  # Windows
cp env.template .env  # macOS/Linux
# Edit .env with your actual credentials

# Run data ingestion (first time only)
python data-ingestion/data_ingestion.py

# Start FastAPI server
python main.py
```

**Using traditional pip:**
```bash
# Install Python dependencies
pip install -r requirements.txt

# Copy environment template and add your API keys
copy env.template .env  # Windows
cp env.template .env  # macOS/Linux
# Edit .env with your actual credentials

# Run data ingestion (first time only)
python data-ingestion/data_ingestion.py

# Start FastAPI server
python main.py
```

Backend runs at: `http://localhost:8001`

### 2. Setup Frontend

Open a **new terminal** window:

```bash
# Navigate to frontend directory
cd rag-chat

# Install dependencies
npm install

# Start Next.js development server
npm run dev
```

Frontend runs at: `http://localhost:3000`

### 3. Test the Application

1. Open browser: `http://localhost:3000`
2. Try asking: "Best budget laptops for students"

---

## Detailed Setup

### Step 1: Clone and Navigate

```bash
cd cssm
```

### Step 2: Setup Python Environment

**Option A: Using `uv` (Recommended)**

```bash
# Install uv
pip install uv

# Create virtual environment and install dependencies
uv venv && uv pip install -r requirements.txt

# Activate
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

**Option B: Using Traditional pip/venv**

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

```bash
# Copy the environment template
copy env.template .env  # Windows
cp env.template .env  # macOS/Linux

# Edit .env file with your actual credentials
notepad .env  # Windows
nano .env  # macOS/Linux
```

**Required Environment Variables:**

```env
# AstraDB Configuration (REQUIRED)
ASTRADB_API_ENDPOINT=https://your-database-id-us-east1.apps.astra.datastax.com
ASTRADB_APPLICATION_TOKEN=AstraCS:your-token-here
ASTRADB_KEYSPACE=your_keyspace_name

# LLM API Keys (at least ONE required)
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**How to get credentials:**

- **AstraDB**: [astra.datastax.com](https://astra.datastax.com)
  - Create free account
  - Create database and keyspace
  - Get API endpoint and token from dashboard

- **API Keys**:
  - Google Gemini: [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
  - Groq: [console.groq.com](https://console.groq.com) (free, fast!)
  - OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
  - Anthropic: [console.anthropic.com](https://console.anthropic.com)

### Step 4: Configure Application

Edit `config/config.yaml`:

```yaml
embedding_model:
  provider: "google"  # or "openai"
  model: "text-embedding-004"

llm_model:
  provider: "google"  # or "groq", "openai", "anthropic"
  model: "gemini-2.5-pro"  # or "llama-3.1-8b-instant" for Groq

retriever:
  top_k: 3  # Number of documents to retrieve

ingestion:
  batch_size: 64
  limit: 1000  # Start with 1000 for testing
  chunk_size: 1200
  chunk_overlap: 150
```

**Recommended for first run:**
- Set `limit: 100` for quick testing
- Use Groq with `llama-3.1-8b-instant` for free, fast responses
- Use Google Gemini for better quality

### Step 5: Run Data Ingestion

```bash
python data-ingestion/data_ingestion.py
```

**Expected output:**
```
INFO | Initializing DataIngestion pipeline
INFO | Loaded JSONL data with 5000000 rows
INFO | Transform will process rows: 100
INFO | Transform complete. Built 100 product entries
INFO | Starting ingestion: total_docs=100, batch_size=64
INFO | Inserted batch 1 (64 docs). Cumulative: 64/100
INFO | Inserted batch 2 (36 docs). Cumulative: 100/100
INFO | Data inserted successfully. Total ids: 100
```

### Step 6: Start Backend Server

```bash
python main.py
```

**Expected output:**
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8001
```

**Test backend:**
```bash
curl http://localhost:8001/health
curl -X POST "http://localhost:8001/retrieve" -H "Content-Type: application/x-www-form-urlencoded" -d "msg=best laptops"
```

### Step 7: Start Frontend

**New terminal:**
```bash
cd rag-chat
npm install
npm run dev
```

**Expected output:**
```
▲ Next.js 15.5.6
- Local:        http://localhost:3000
```

---

## Using uv Package Manager

`uv` is an extremely fast Python package manager (10-100x faster than pip).

### Installation

```bash
# Install uv
pip install uv

# Or official installer (recommended)
# Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Common Commands

```bash
# Create venv and install dependencies
uv venv && uv pip install -r requirements.txt

# Activate venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install packages
uv pip install package-name
uv pip install -r requirements.txt

# Run scripts
uv run python main.py

# List packages
uv pip list

# Update packages
uv pip install --upgrade package-name
```

### Speed Comparison

| Operation | pip | uv | Speedup |
|-----------|-----|-----|---------|
| Install 100 packages | ~45s | ~2s | **22x faster** |
| Resolve dependencies | ~10s | ~0.5s | **20x faster** |
| Create venv | ~2s | ~0.1s | **20x faster** |

---

##  Configuration

### Environment Variables

Create `.env` from template:

```bash
copy env.template .env  # Windows
cp env.template .env  # macOS/Linux
```

Required variables:
- `ASTRADB_API_ENDPOINT` - Your AstraDB endpoint
- `ASTRADB_APPLICATION_TOKEN` - Your AstraDB token
- `ASTRADB_KEYSPACE` - Your keyspace name
- At least one LLM API key (Google, Groq, OpenAI, or Anthropic)

### Application Configuration

Edit `config/config.yaml`:

```yaml
data:
  json_path: "data/merged_electronics_data.json"
  jsonl_path: "data/merged_electronics_data.jsonl"

astradb:
  collection_name: "cssm"

embedding_model:
  provider: "google"
  model: "text-embedding-004"

llm_model:
  provider: "google"
  model: "gemini-2.5-pro"

retriever:
  top_k: 3
  top_p: 0.95

ingestion:
  batch_size: 64
  max_retries: 6
  backoff_initial_seconds: 1.0
  chunk_size: 1200
  chunk_overlap: 150
  limit: 100000
  shuffle: false
```

---

## API Endpoints

### GET `/health`

Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "service": "Amazon Product Assistant API"
}
```

### POST `/retrieve`

Chat endpoint for product queries

**Request:**
```bash
curl -X POST "http://localhost:8001/retrieve" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "msg=best laptops for students"
```

**Response:**
```json
{
  "response": "Based on reviews, here are the best laptops..."
}
```

---

## Frontend Setup

### Installation

```bash
cd rag-chat
npm install
npm run dev
```

### Features

**Amazon-style UI** - Dark header with orange accents  
**Real-time Chat** - Instant messaging interface  
**Typing Indicators** - Shows bot is thinking  
**Message Timestamps** - All messages have timestamps  
**Markdown Cleaning** - Removes **bold** and *italic* formatting  
**Auto-scroll** - Automatically scrolls to latest message  
**Responsive Design** - Works on mobile and desktop  
**Error Handling** - Graceful error messages  

### Customization

**Change Backend URL:**
Edit `src/components/ChatInterface.tsx`:
```typescript
const response = await fetch('http://localhost:8001/retrieve', {
  // Change this URL to your backend
```

**Change Colors:**
```typescript
// Orange gradient (Amazon's brand color)
bg-gradient-to-r from-[#ff9900] to-[#ff6b00]

// Dark header (Amazon's header color)
bg-gradient-to-b from-[#131921] to-[#232f3e]
```

---

## Troubleshooting

### Backend Issues

**Module not found:**
```bash
pip install -r requirements.txt
# or
uv pip install -r requirements.txt
```

**Environment variable missing:**
- Check `.env` file exists
- Verify all required variables are set
- Ensure no extra spaces in values

**Database timeout:**
- Check data ingestion completed successfully
- Verify AstraDB credentials
- Reduce batch size in config

**No results found:**
- Ensure data ingestion ran successfully
- Check if collection has documents
- Verify embedding model is correct

### Frontend Issues

**CORS error:**
- Ensure FastAPI CORS middleware is enabled
- Check backend is running on port 8001
- Verify CORS origins include http://localhost:3000

**Connection refused:**
- Verify backend is running on port 8001
- Check firewall settings
- Confirm backend URL in ChatInterface.tsx
- Test backend: `curl http://localhost:8001/health`

**Icons not showing:**
```bash
npm install lucide-react
```

### Common Commands

```bash
# Test backend health
curl http://localhost:8001/health

# Test chat endpoint
curl -X POST "http://localhost:8001/retrieve" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "msg=best laptops"

# Check if frontend is running
curl http://localhost:3000
```

---

## Project Structure

```
cssm/
├── config/
│   ├── config.yaml              # Configuration
│   └── config_loader.py         # Config loader
├── data/
│   └── merged_electronics_data.jsonl  # Product data (5M reviews)
├── data-ingestion/
│   └── data_ingestion.py        # Data pipeline
├── retriever/
│   └── retrieval.py             # Vector search
├── prompts/
│   └── prompt.py                # LLM prompts
├── utils/
│   └── model_loader.py          # Model initialization
├── templates/
│   └── chat.html                # HTML template (legacy)
├── static/
│   └── style.css                # Styles (legacy)
├── main.py                      # FastAPI server
├── requirements.txt             # Python dependencies
├── env.template                 # Environment template
├── .gitignore                   # Git ignore rules
└── rag-chat/                    # Next.js frontend
    ├── src/
    │   ├── app/
    │   │   ├── page.tsx         # Main page
    │   │   ├── layout.tsx       # Root layout
    │   │   └── globals.css      # Global styles
    │   └── components/
    │       └── ChatInterface.tsx # Chat component
    └── package.json             # Node dependencies
```

---

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **LangChain** - RAG orchestration
- **AstraDB** - Vector database
- **Google Gemini** - LLM for responses
- **Google Embeddings** - Text embeddings

### Frontend
- **Next.js 15** - React framework
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS
- **Lucide React** - Beautiful icons

---

##  Features

### Backend
-  RAG pipeline with retrieval and generation
-  Batch data ingestion with retries
-  Configurable chunking and embeddings
-  Error handling and logging
-  CORS enabled for frontend
-  Health check endpoint

### Frontend
-  Amazon-style UI design
-  Real-time chat interface
-  Typing indicators
-  Message timestamps
-  Auto-scroll to latest message
-  Markdown cleaning
-  Error handling
-  Responsive design

---

##  Usage Examples

### Query Examples:
- "Best budget laptops for students"
- "Laptops with good battery life"
- "Lenovo laptops for gaming"
- "Fast performance laptops"
- "Reliable laptops for work"

### Example Response:
```
Based on the reviews, here are the best budget laptops for students:

1. **Acer Aspire 5** (4.5/5 stars) - Students love this laptop for its 
   affordable price and decent performance. Many reviewers mention it handles 
   multiple browser tabs and Microsoft Office smoothly.

2. **HP Pavilion 15** (4.3/5 stars) - Great for students who need a reliable 
   laptop for online classes and assignments. Reviewers praise the keyboard 
   and battery life.

I'd recommend the Acer Aspire 5 if you're looking for the best value for money, 
or the HP Pavilion 15 if you prioritize build quality and battery life.
```

---

##  Scaling Up

Once everything works with 100 records:

1. **Increase data ingestion:**
   ```yaml
   # In config/config.yaml
   ingestion:
     limit: 10000  # or more
   ```

2. **Run ingestion again:**
   ```bash
   python data-ingestion/data_ingestion.py
   ```

3. **Monitor performance:**
   - Check AstraDB dashboard for document count
   - Test query response times
   - Adjust `top_k` for better results

---

##  Project Status

### Fully Functional 

**Architecture:**
-  Backend API server (FastAPI on port 8001)
-  Frontend UI (Next.js on port 3000)
-  RAG pipeline (retrieval + generation)
-  Vector database integration (AstraDB)
-  CORS configuration for frontend
-  Health check endpoint
-  Complete documentation
-  Environment variable template
-  Proper dependency management
-  Git ignore for large files

**Recent Improvements:**
-  Added `uv` package manager support (10-100x faster)
-  Fixed duplicate `load_dotenv()` in main.py
-  Added missing `pyyaml` dependency
-  Updated port from 8000 to 8001
-  Added comprehensive documentation
-  Added health check endpoint
-  Excluded data files from git

---

##  Data Stats

- **Total Records**: 5,000,000
- **Laptop Records**: 397,334 (7.9%)
- **Categories**: Electronics, Computers, Cell Phones, Camera & Photo, etc.

---

##  Contributing

Contributions welcome! Please open an issue or submit a PR.

---

##  License

MIT License

---

##  Citation

If you use this project or the Amazon Reviews dataset in your research, please cite:

**Amazon Reviews Dataset:**
```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

**This Project:**
```bibtex
@software{amazon_product_assistant_2024,
  title={Amazon Product Assistant - RAG-based Customer Support Bot},
  author={Gowtham Arulmozhi},
  year={2025},
  url={https://github.com/wothmag07/cssm}
}
```

---

## Support

For issues or questions:
- Check the [Troubleshooting](#troubleshooting) section
- Review error logs in terminal
- Check AstraDB dashboard for data status
- Verify all environment variables are set correctly

---

Happy coding!
