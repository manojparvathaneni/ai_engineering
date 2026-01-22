# AI Engineering Projects Overview

This repository contains five hands-on projects completed over five weeks. The projects build capability with large language models, retrieval, tool use, research workflows, and multimodality. Week six is a capstone where you design your own system, tool, or startup idea based on your learnings.

Each week, the new project folder is released with:
- a notebook (primary workflow)
- supporting data
- an `environment.yml` for reproducible setup

## Project index

| Week | Project | What you build | Main topics |
|---|---|---|---|
| 1 | LLM Playground | Notebook playground for prompting + decoding | tokens, decoding, prompt patterns |
| 2 | Customer Support Chatbot | RAG chatbot + small UI | chunking, embeddings, retrieval, RAG |
| 3 | Ask-the-Web Agent | Tool-using web QA agent | tool calling, planning loop, schemas |
| 4 | Deep Research System | Multi-step research workflow | evidence gathering, synthesis, multi-agent |
| 5 | Multimodal Agent | Text + image/video generation agent + UI | routing, image gen, video gen |
| 6 | Capstone | Your own system | design + build + demo |

Release Schedule: Each week, a new project folder is added to this repo containing the notebook, data, and environment file.

## Repo structure

Each week has its own folder:
- `project_1/`
- `project_2/`
- `project_3/`
- `project_4/`
- `project_5/`

Inside each folder you will typically find:
- `project.ipynb`
- `environment.yml` (or `requirements.txt`)
- optional: `data/`
- optional: `app/` or `streamlit.py` for demos


## Quick start

### Step 1: Get the code

```bash
git clone <REPO_URL>
cd <REPO_NAME>
```
When a new week is released:
```bash
git pull
```

### Step 2: Open the project in your IDE
Open your IDE of choice (VS Code, Cursor, Antigravity), then open the repo folder <REPO_NAME>.

### Step 3: Work on the current week
Open the instructions for the current week in the repo (usually the notebook inside a `project_X/` folder), then follow the steps in order.

You can run the projects either on **Google Colab** (no local setup required) or **locally** (using Conda environments for reproducibility or uv).

#### Option A: Run in Google Colab
1. Open the notebook for the current week (upload to Colab or open via link if provided).
2. Install dependencies if needed.
3. Set any required API keys if needed.
4. Adjust file paths for Colab (use /content/...).

#### Option B: Run locally with Conda
Each project includes an `environment.yml` to keep dependencies consistent.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download).
2. Create and activate the environment from the provided YAML file:
   ```bash
   conda env create -f environment.yml
   conda activate <ENV_NAME>
   ```
   The environment name is set inside the YAML. You can change it if desired.
3. Launch Jupyter and open the notebook for the current week:
   ```bash
   jupyter notebook
   ```

#### Option C: Run locally with uv (Blazing Fast)
uv is a modern Python package installer written in Rust. It is significantly faster than Conda or Pip and acts as a drop-in replacement. If you find Conda slow to solve environments, use this.
1. Install uv (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Create a requirements.txt file and list all dependecies for that project. You can find these in the provided `environment.yml`
2. Create a virtual environment and install dependencies:
   ```bash
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

**Recommendation:** Use Colab for projects 1 and 5, and local development for projects 2, 3, and 4.



## Project expectations
- Each project is a guided template with sections marked “your code here”.
- There are multiple valid implementations. You can deviate from the default stack and experiment.
- No submission is required.
- In live sessions, we will walk through one reference implementation and discuss trade-offs.


## Getting Help
Use the cohort Q/A space for questions and debugging. You are also welcome to share your thoughts, opinions, and interesting findings in the same space.  

## Weekly projects

### Project 1: Build an LLM Playground
An introductory project to explore how prompts, tokenization, and decoding settings work in practice, building the foundation for effective use of large language models.

**Learning objectives:**
- Tokenization of raw text into discrete tokens
- Basics of GPT-2 and Transformer architectures
- Loading pre-trained LLMs with Hugging Face
- Decoding strategies for text generation
- Completion vs. instruction-tuned models

### Project 2: Customer-Support Chatbot for an E-Commerce Store
A hands-on project to build a retrieval-based chatbot that answers customer questions for an imaginary e-commerce store.

**Learning objectives:**
- Ingest and chunk unstructured documents
- Create embeddings and index with FAISS
- Retrieve context and design prompts
- Run an open-weight LLM locally with Ollama
- Build a RAG (Retrieval-Augmented Generation) pipeline
- Package the chatbot in a minimal Streamlit UI

### Project 3: Ask-the-Web Agent
A project to create a simplified Perplexity-style agent that searches the web, reads content, and provides answers.

**Learning objectives:**
- Understand why tool calling is useful for LLMs
- Implement a loop to parse model calls and execute Python functions
- Use function schemas (docstrings and type hints) to scale across tools
- Apply LangChain for function calling, reasoning, and multi-step planning
- Combine Llama-3 7B Instruct with a web search tool to build an ask-the-web agent

### Project 4: Build a Deep Research System
A project focused on reasoning workflows, where you design a multi-step agent that plans, gathers evidence, and synthesizes findings.

**Learning objectives:**
- Apply inference-time scaling methods (zero-shot/few-shot CoT, self-consistency, sequential decoding, tree-of-thoughts)
- Gain intuition for training reasoning models with the STaR approach
- Build a deep-research agent that combines reasoning with live web search
- Extend deep research into a multi-agent system

### Project 5: Build a Multimodal Generation Agent
A project to build an agent that combines textual question answering with image and video generation capabilities within a unified system.

**Learning objectives:**
- Generate images from text using Stable Diffusion XL
- Create short clips with a text-to-video model
- Build a multimodal agent that handles questions and media requests
- Develop a simple Gradio UI to interact with the agent

### Week 6: Capstone Project

Design your own system! Use the patterns from Weeks 1-5 to build a prototype, internal tool, or research workflow.

## Reference docs and readings

The following documentation pages cover the core libraries and services used across projects:
- [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html): Manage isolated Python environments and dependencies with Conda  
- [Pip documentation](https://pip.pypa.io/en/stable/user_guide/): Install and manage Python packages with pip  
- [duckduckgo-search](https://pypi.org/project/duckduckgo-search/): Python library to query DuckDuckGo search results programmatically  
- [gradio](https://www.gradio.app/guides): Build quick interactive demos and UIs for machine learning models  
- [Streamlit documentation](https://docs.streamlit.io/): Build and deploy simple web apps for data and ML projects  
- [huggingface_hub](https://huggingface.co/docs/huggingface_hub/index): Access and share models, datasets, and spaces on Hugging Face Hub  
- [langchain](https://python.langchain.com/docs/get_started/introduction): Framework for building applications powered by LLMs with memory, tools, and chains  
- [numpy](https://numpy.org/doc/stable/): Core library for numerical computing and array operations in Python  
- [openai](https://platform.openai.com/docs): Official API docs for using OpenAI models like GPT and embeddings  
- [tiktoken](https://github.com/openai/tiktoken): Fast tokenizer library for OpenAI models, used for counting tokens  
- [torch](https://pytorch.org/docs/stable/index.html): PyTorch deep learning framework for training and running models  
- [transformers](https://huggingface.co/docs/transformers/index): Hugging Face library for using pre-trained LLMs and fine-tuning them  
- [llama-index](https://docs.llamaindex.ai/en/stable/): Data framework for connecting external data sources to LLMs  
- [chromadb](https://docs.trychroma.com/): Open-source vector database for storing and retrieving embeddings in RAG systems  



