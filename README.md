# Ready Tensor Agentic AI Certification - Week 3

This repository contains the practical code and exercises for **Week 3** of the Ready Tensor Agentic AI Certification program. Week 3 builds on the modular prompt engineering concepts introduced in Week 2 and adds practical examples of multi-turn conversations and robust system prompts.

## What You'll Learn

**Lesson 1 - Basic LLM Calls & Multi-Turn Conversations:**

- Make your first LLM calls using LangChain and Groq.
- Ground your questions in publication content for accurate, context-aware answers.
- Explore how context is managed in multi-turn conversations.

**Lesson 2 - System Prompts for Control & Safety:**

- Craft modular system prompts that define your assistant's behavior, boundaries, and tone.
- Learn how to handle manipulative questions and safeguard your assistant's instructions.
- See how small changes in your system prompt dramatically improve reliability and security.

You'll experiment with real-world scenarios, using a publication about Variational Autoencoders (VAEs) as the test case.

## Repository Structure

```

rt-agentic-ai-cert-week3/
├── code/
│   ├── config/
│   │   ├── config.yaml                 # App config
│   │   └── prompt_config.yaml          # Prompt configurations for system prompt examples
│   ├── paths.py                        # File path configurations
│   ├── prompt_builder.py               # Modular prompt construction functions
│   ├── run_wk3_l1_example_1_2.py       # Lesson 1: Basic LLM calls and publication grounding
│   ├── run_wk3_l1_example_3.py         # Lesson 1: Interactive terminal chat example
│   ├── run_wk3_l2_sys_prompt_example.py # Lesson 2: System prompt examples and testing
│   ├── run_wk3_l3a_memory_strategies.py # Lesson 3A: Memory strategies comparison
│   ├── run_wk3_l4_vector_db_ingest.py  # Lesson 4: Vector DB ingestion script
│   ├── run_wk3_l4_vector_db_rag.py     # Lesson 4: Vector DB RAG example
│   └── utils.py                        # Utility functions
├── data/                               # Sample publications for exercises
│   ├── 57Nhu0gMyonV.md
│   ├── ljGAbBceZbpv.md
│   ├── tum5RnE4A5W8.md
│   ├── yzN0OCQT7hUS-sample-questions.yaml  # Sample questions used in lesson 3 exercise
│   └── yzN0OCQT7hUS.md
├── outputs/                            # Generated prompts and LLM responses
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

```

## Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/readytensor/rt-agentic-ai-cert-week3.git
   cd rt-agentic-ai-cert-week3
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Groq API key:**

   Create a `.env` file in the root directory and add your API key:

   ```
   GROQ_API_KEY=your-api-key-here
   ```

   You can get your API key from [Groq](https://console.groq.com/).

---

## Running the Lessons

This repository includes scripts to help you experiment with different types of interactions and prompt configurations.

### Lesson 1 — Basic LLM Calls & Multi-Turn Conversations

- **`run_wk3_l1_example_1_2.py`**

  - **Example 1:** A simple LLM call answering a general question about VAEs.
  - **Example 2:** The same question, but grounded in the publication content.

- **`run_wk3_l1_example_3.py`**

  - An **interactive terminal-based chat** with the assistant, showing how multi-turn conversations work and how context is managed.

### Lesson 2 — System Prompts for Control & Safety

- **`run_wk3_l2_sys_prompt_example.py`**

  - Tests system prompt configurations (basic and advanced) to see how they handle manipulative questions and maintain professional tone and safety.

### Lesson 3A — Memory Management Strategies

- **`run_wk3_l3a_memory_strategies.py`**
  - Compares three memory management strategies: stuffing everything, trimming to recent messages, and summarizing conversation history.
  - Simulates a long conversation using real questions, saving detailed results (Q&A pairs, token usage, and final prompts) in the `outputs/` directory.
  - Includes an interactive mode for running a single strategy or a full comparison report.

### Lesson 4 — Vector Database & RAG Implementation

- **`run_wk3_l4_vector_db_ingest.py`**

  - **Vector Database Ingestion:** Initializes a ChromaDB instance with persistent storage, chunks publications into smaller documents, embeds them using HuggingFace transformers, and stores them in the vector database for semantic search.

- **`run_wk3_l4_vector_db_rag.py`**

  - **Retrieval-Augmented Generation (RAG):** Interactive terminal-based chat that retrieves relevant documents from the vector database based on user queries and generates contextual responses using retrieved content. Includes configurable similarity thresholds and result counts.

Each script saves outputs and transcripts in the `outputs/` directory for easy review and comparison.

---

## Key Features

- **System Prompt Testing:** Explore how different system prompt configurations affect your assistant’s behavior, style, and security.
- **Multi-Turn Conversations:** See how context builds up in an interactive chat, and how to manage it effectively.
- **Real-World Examples:** Grounded in a research publication about Variational Autoencoders (VAEs) to simulate real user scenarios.
- **Modular Approach:** Use the same modular components (role, tone, constraints) from Week 2, but applied in system prompts for reliability and trustworthiness.

---

## How It Works

The `prompt_builder.py` module takes YAML configurations and dynamically assembles system prompts. You can:

- Adjust modular components like **role, tone, output constraints, and format**.
- Test how small changes improve your assistant's reliability, especially for edge cases and manipulative questions.
- Use real-world scenarios to refine your system prompts and see their impact.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Contact

**Ready Tensor, Inc.**

- Email: contact at readytensor dot com
- Issues & Contributions: Open an issue or pull request on this repository
- Website: [Ready Tensor](https://readytensor.com)
