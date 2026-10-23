# Agentic AI RFP Evaluation Platform

A Streamlit application for evaluating vendor proposals against configurable RFP
criteria. It uses LangChain with selectable AI providers for structured
evaluations, PyMuPDF to read proposal PDFs, SQLite for local persistence, and
Plotly for leaderboard analytics.

## Requirements

- Python 3.10 or newer
- An API key for one of the supported AI providers: OpenAI, Anthropic, Google
  Gemini, Groq, or Mistral
- A text-based PDF proposal (scanned/image-only PDFs need OCR first)

## Windows setup (PowerShell)

Run these commands from the project directory:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, allow it for the current terminal and activate
again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Enter your provider API key in the **AI provider authentication** panel in the
app's left sidebar, then select **Validate key and load models**. After
validation, choose an available LLM model from the model dropdown. The key is
held only in the active Streamlit session and is not saved to the database or
`.env`; re-enter and validate it in a new session.

## Run the app

With the virtual environment activated, run:

```powershell
streamlit run app.py --server.address 127.0.0.1
```

Streamlit prints a local URL, normally `http://localhost:8501`.

## Features

- Upload multiple PDF proposals and evaluate them against the RFP context you
  provide. Vendor names are inferred from filenames such as
  `Apex_Systems_RFP_Proposal.pdf` → `Apex Systems`.
- Authenticate against OpenAI, Anthropic, Google Gemini, Groq, or Mistral and
  load the models available to your API key.
- Score criteria with weighted totals and short, verifiable proposal quotations.
- Compare vendors using a leaderboard, Peer Performance Index (PPI), and a
  configurable industry benchmark index. Both views include every proposal
  evaluation saved in the local database, including completed batch uploads,
  while filtering duplicate proposal records.
- Browse all vendor evaluation results and evidence, with a selectable detail
  view for each proposal.
- Clear all saved proposal results and the current upload selection using the
  confirmed sidebar **Clear uploads and results** action. Criteria and benchmark
  settings are preserved.
- Add, remove, or edit evaluation criteria while keeping weights at 100%.
- Store evaluations, criteria, and the benchmark in `rfp_evaluations.db` in the
  project directory.

Name proposals with the vendor name followed by `_RFP_Proposal.pdf` (for example,
`BrightPath_Tech_RFP_Proposal.pdf`). Spaces and hyphens are accepted as
separators too. Each proposal is evaluated and saved independently, so a failed
upload does not prevent the other proposals from being processed.
Identical PDF files are detected by SHA-256 and skipped both within the current
upload batch and on later uploads. For older saved evaluations without a file
fingerprint, a matching vendor and filename is treated as a duplicate.

The default criteria are Technical Capability (25%), Relevant Experience (20%),
Delivery Approach (15%), Security & Compliance (15%), Cost & Commercial Value
(15%), and Support & Risk Management (10%). Evaluation results retain a snapshot
of the criteria and weights used at evaluation time.

The app does not send proposal text to an LLM until you submit an evaluation.
Proposals are sent to the configured model provider for that evaluation.
