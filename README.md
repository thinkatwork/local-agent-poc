# Local Document Agent POC

A small local AI agent with a browser-based chat UI. It uses Ollama for inference and can only search or read `.txt`, `.md`, and text-based `.pdf` files in one configured folder. It cannot edit documents, run shell commands, or access other folders through its tools.

## Requirements

- macOS 14 or later
- Ollama for macOS
- Python 3.10 or later
- Internet access for the initial app/model/dependency downloads; model inference then runs through the local Ollama service

## Run on a Mac

1. Install and open Ollama: <https://ollama.com/download/mac>
2. In Terminal, download the model:

   ```bash
   ollama pull qwen3:8b
   ```

3. In Terminal, enter the folder containing this repo and run:

   ```bash
   cd local-agent-poc
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   streamlit run app.py
   ```

4. Streamlit opens the chat UI in your browser, normally at `http://localhost:8501`.
5. Use the sidebar to choose a documents folder or upload a few test files. Ask questions in the chat box. Expand the tool-call panel to see which documents the agent searched or read.

If your Mac has limited free memory, try `ollama pull qwen3:4b` and enter `qwen3:4b` in the model field.

## Publish this project to GitHub

1. Sign in to GitHub and create a **new empty repository** named `local-agent-poc`. Choose **Private** while you are experimenting. Do not ask GitHub to add a README, `.gitignore`, or license; those files are already in this project.
2. On your Mac, open Terminal and go to this project folder. One easy way is to type `cd ` (including the space), drag the `local-agent-poc` folder from Finder into Terminal, then press Return.
3. Run these commands, replacing `YOUR-GITHUB-USERNAME` with your GitHub username:

   ```bash
   git init -b main
   git add .
   git commit -m "Initial local document agent POC"
   git remote add origin https://github.com/YOUR-GITHUB-USERNAME/local-agent-poc.git
   git push -u origin main
   ```

4. If GitHub asks you to authenticate, complete its sign-in prompt. When the push finishes, refresh the repository page.

The `.gitignore` excludes your virtual environment, temporary Python files, and documents added to `documents/`; the fictional `example-handbook.md` remains included. Keep personal or work documents out of GitHub. This project has no license file; decide on a license before making the repository public if you want others to reuse the code.

## What the agent does

The model receives the conversation and three function descriptions. It may request `list_documents` to see filenames, `search_documents` to find relevant passages, and `read_document` to inspect one matching file. Python checks the file path and performs the read-only operation. The returned text goes back to the model, which drafts the answer and cites the source filename.

The tool loop stops after five rounds. Files are limited to 5 MB each and document text is capped before being sent to the model. PDFs must contain selectable text; scanned PDFs need OCR first. Content in documents is treated as reference material, not instructions.

## POC boundaries

- Only `.txt`, `.md`, and `.pdf` files are listed, searched, or read.
- Each file is limited to 5 MB; extracted text is capped before it is sent to the model.
- The tools only read files under the configured folder.
- No email, web browsing, shell, database, or write actions are connected.
- The local web UI is intended for use on your own Mac. Do not expose its port to your network without adding authentication and other controls.

## Suggested first evaluation

Try ten questions: four with direct answers, three requiring multiple passages, and three whose answers are absent. Check that the agent names the right source and says when it cannot find support. Use a throwaway set of non-sensitive files for the first run.
