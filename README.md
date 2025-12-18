# tunedd-backend-challenge
# Local RAG Server

A local Retrieval-Augmented Generation server using Google's ADK, LiteLLM and Qdrant.

### Models

This server uses ```nomic-embed-text``` for text embeddings and ```gemma3:1b``` as a chat model, but the RAG server is model-agnostic, so feel free to change the ```.env``` file to fit other models to your taste. Only make sure the provided model name is available in Ollama!

## Prerequisites

- Docker and Docker Compose
- Python 3.13+ (with pyenv or equivalent)
- Poetry (for Python dependency management)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/mpagner/tunedd-backend-challenge-main
cd tunedd-backend-challenge-main

# Copy environment variables
cp .env.example .env

# Edit .env with your preferred models
vim .env

# Start infrastructure (Ollama, Qdrant database)
docker-compose up -d

# Download Poetry (skip if you already have it installed)
pipx install poetry

# In project root folder, set up Python environment
poetry install
poetry shell

# application
python -m main

# ADK
adk web . #interact with web GUI
adk run . #for console
```

If you have a GPU, you can run a docker image to help Ollama detect it and run the models locally. Scripts are available for both Nvidia and AMD GPUs--choose according to your brand and run ```docker compose -f docker-compose.{brand}.override.yml```.
NOTE: on AMD, you might get warnings from Ollama asking to install further ROCm packages in order to run models.

Done! You can now ask about AI-related documents--here's some examples to test:

```bash
What do Asadi et al. talk about in their article?
What are pending challenges in the field of LLM Multi-agent systems?
```
# Known issues

What follows is a list of known bugs in the project, the causes of which are being investigated:

- The agent tends to return chat responses in JSON format instead of plain text.
- The agent might on occasion mistakenly call a non-existent function like ```function_rag_search_tool()```, ```assistant``` or other kinds of names. For now, a possible workaround is to send the text prompt again, or start a new session from the UI.

# Common Issues
- Port conflicts: Check if ports 11434, 6333, 6334, 8000 are free

- Model download failures: Check internet connection and model names

- Memory issues: Use smaller models or increase Docker memory limits
