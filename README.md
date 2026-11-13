# Research Assistant POC

A small, user-directed research assistant with a browser-based chat UI. The UI and Python app run on your Mac. You choose where the model runs:

- **Ollama (default):** the model runs locally on your Mac.
- **Gemini:** the model runs at Google, over the internet.

It has two modes:

- **Web research:** a short, fixed workflow. It searches the public web with Tavily, reads a few public pages, writes evidence notes, optionally reviews them, then writes a cited answer. Works with Ollama or Gemini.
- **Local documents:** searches and reads `.txt`, `.md` and text-based `.pdf` files in one folder. **Ollama only**, so document contents never leave your Mac.

It is read-only. It cannot edit documents, run shell commands, access other folders, sign in to websites, submit forms, post or send email. It is a research aid that runs a short, controlled workflow, not an autonomous agent. The reviewer step can catch some weak claims but does not guarantee correctness: check important answers against the cited sources.

## How it fits together

```
Browser on your Mac ──▶ Streamlit + Python app (on your Mac)
                          ├─▶ Model: Ollama on this Mac   OR   Gemini at Google (internet)
                          ├─▶ Tavily: receives search text, returns result links/snippets
                          └─▶ Public web pages: fetched directly by the app (read-only GET)
```

- **"The UI is local" does not mean "all processing is local".**
- **Gemini mode:** your question, the search results and excerpts of the fetched web pages are sent to Google. Search queries go to Tavily. Local documents are not sent (Local documents mode is disabled with Gemini).
- **Ollama mode:** model inference stays on your Mac. Web research still sends search queries to Tavily and fetches pages from the public internet.
- **Local documents mode with Ollama:** fully local apart from the model download.

## Requirements

- macOS 14 or later, Python 3.10 or later
- For Ollama mode: Ollama for macOS
- For web research: a Tavily API key
- For Gemini mode: a Gemini API key from Google AI Studio

## Run on a Mac

1. (Ollama mode) Install and open Ollama: <https://ollama.com/download/mac>, then download the model:

   ```bash
   ollama pull qwen3:8b
   ```

   If your Mac has limited free memory, use `qwen3:4b` and set `OLLAMA_MODEL=qwen3:4b` in `.env`.

2. In Terminal, in the repository folder:

   ```bash
   cd research_agent
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   open -e .env      # add your keys here — never in a chat or in source code
   git check-ignore .env && echo ".env is ignored by Git"
   streamlit run app.py
   ```

3. Streamlit opens the chat UI at `http://localhost:8501`. The sidebar shows the model provider, the mode, the reviewer toggle and the per-question limits.

### Switching model provider

- `.env` sets the default: `MODEL_PROVIDER=ollama` or `MODEL_PROVIDER=gemini`. Restart the app after editing `.env`.
- You can also switch for the current session in the sidebar.
- The app never switches provider on its own. If the chosen provider is misconfigured or fails, it shows an error and stops.

## Settings (`.env`)

| Setting | Default | Meaning |
|---|---|---|
| `MODEL_PROVIDER` | `ollama` | `ollama` or `gemini` |
| `OLLAMA_MODEL` | `qwen3:8b` | Local model name |
| `GEMINI_API_KEY` | blank | Your Gemini API key |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Gemini model ID |
| `GEMINI_TIER` | `free` | `free` or `paid` (see below) |
| `GEMINI_INPUT_PRICE_PER_M` / `GEMINI_OUTPUT_PRICE_PER_M` | `0.25` / `1.50` | USD per 1M tokens, used only for estimates |
| `MAX_ESTIMATED_SPEND_PER_QUESTION` | blank | Optional estimated USD cap per question; required when `GEMINI_TIER=paid` |
| `TAVILY_API_KEY` | blank | Your Tavily API key |
| `MAX_MODEL_CALLS_PER_QUESTION` | `3` | Hard cap on model calls (1–10) |
| `MAX_SEARCHES_PER_QUESTION` | `3` | Hard cap on Tavily searches (0–5) |
| `MAX_FETCHES_PER_QUESTION` | `4` | Hard cap on fetched pages (0–8) |
| `MAX_OUTPUT_TOKENS_PER_CALL` | `1024` | Output-token cap per model call |
| `MAX_INPUT_CHARS_PER_CALL` | `40000` | Text sent per model call is cut to this length |
| `MODEL_REQUEST_TIMEOUT_SECONDS` | `60` | Timeout per model request |

Invalid values are shown as errors in the sidebar, not silently replaced by another provider.

## What the web-research workflow does

1. **Query planning (only when the call limit leaves room):** if `MAX_MODEL_CALLS_PER_QUESTION` is higher than the calls the workflow needs, the model writes up to `MAX_SEARCHES_PER_QUESTION` search queries. Otherwise your question is used as the search query.
2. **Search:** Python sends the queries to Tavily (basic search) and stops at the search limit.
3. **Read pages:** Python picks the top results, preferring different sites, and fetches up to the page limit itself. Tavily's extract and crawl features are never used.
4. **Researcher (1 model call):** writes short evidence notes with title, URL and date.
5. **Reviewer (1 model call, optional):** flags unsupported, conflicting, dated or weak claims. It does no extra searching.
6. **Writer (1 model call):** writes the answer, citing only pages that were actually fetched.

With the defaults this is 3 model calls with the reviewer on, or 2 with it off (and 3 if the planner fits). The model never gets tools: it only reads the text Python gives it, so it cannot search, browse, or act on anything by itself.

Under each answer the app shows:

- a **workflow trace**: every search, fetch and model step, with its input size and output
- a **usage line**: model calls, input/output/thinking tokens, Tavily searches, pages fetched, and for Gemini an estimated cost

## Hard limits enforced in code

These are enforced in Python, not just requested in prompts:

- maximum model calls, searches and fetched pages per question
- maximum output tokens per call, and text sent per call cut to `MAX_INPUT_CHARS_PER_CALL`
- request timeout
- **no retries for Gemini requests:** the SDK is set to 1 attempt, and multi-hop continuation is turned off
- a quota or rate-limit error (HTTP 429) stops the question
- optional estimated spend cap: before each Gemini call, a worst-case estimate is checked against `MAX_ESTIMATED_SPEND_PER_QUESTION`, and the call is not sent if it could exceed it
- no automatic fallback between providers
- Google Search grounding and paid Tavily endpoints are not used anywhere in the code

**These are application limits, not a bill guarantee.** Cost figures are estimates from the prices in `.env`. Token counts come from the provider when it reports them; when it doesn't, the app estimates them and says so.

## Set up Google Gemini (cloud model)

1. Open [Google AI Studio API keys](https://aistudio.google.com/apikey) and sign in with a Google account.
2. Create or select a project and create an API key. For a new user, AI Studio may create a project and key automatically.
3. **Start on the free tier.** `gemini-3.1-flash-lite` was listed with a free tier on Google's pricing page on 2026-11-13. Don't link a billing account just to test the app.
   - Free-tier rate limits apply ([rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)).
   - Google's pricing page says free-tier content **is used to improve Google's products**, and paid-tier content is not. Don't send personal, confidential, client, work or family information on the free tier.
   - Review Google's current [Gemini API terms](https://ai.google.dev/gemini-api/terms) yourself.
4. On your Mac, in the repository folder, run `cp .env.example .env` if you haven't already. Open `.env` and paste the key after `GEMINI_API_KEY=`.
   - Don't paste the key into Claude, any other chat, or source code.
   - Check that `git check-ignore .env` prints `.env` before you make any commit.
5. Set `MODEL_PROVIDER=gemini` and keep `GEMINI_MODEL=gemini-3.1-flash-lite`, the model the code was set up for. To go back to local inference, set `MODEL_PROVIDER=ollama` or pick Ollama in the sidebar.
6. You don't need Ollama running in Gemini mode, and `qwen3:8b` is not loaded into your Mac's memory. The UI and Python app still run locally.
7. Restart the app, select **Web research**, and ask one short, non-sensitive question. Check the usage line under the answer and the usage page in Google AI Studio.
8. **Paid use** requires linking Google Cloud Billing to the project. Google's billing guide has described a minimum prepayment (for example $5 or equivalent) for some setup flows; check the current prompts before you proceed ([billing](https://ai.google.dev/gemini-api/docs/billing)). To use paid mode in this app, set:
   - `GEMINI_TIER=paid`
   - both prices
   - `MAX_ESTIMATED_SPEND_PER_QUESTION`, for example `0.05`
9. **Set a Google Cloud spend cap before paid use.** In the Google Cloud console, go to **Billing → Budgets & alerts**, create a budget scoped to the project that holds your Gemini API key, set a low amount, and choose the option that caps spend rather than only alerting.
   - An **alert-only budget is not a spending limit**: it emails you, and usage continues.
   - A **spend cap** can stop eligible services, but Google documents that enforcement can be delayed, and requests already in flight can still finish and be billed.
   - Keep the app's own limits on either way.
   - See [spend caps](https://docs.cloud.google.com/billing/docs/how-to/budgets-spend-caps) and [budgets and alerts](https://docs.cloud.google.com/billing/docs/how-to/budgets), and check which services are eligible.
10. Immediately before enabling paid use, re-check these pages: [pricing](https://ai.google.dev/gemini-api/docs/pricing) · [billing](https://ai.google.dev/gemini-api/docs/billing) · [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) · [spend caps](https://docs.cloud.google.com/billing/docs/how-to/budgets-spend-caps).

**The free-tier setting cannot see your billing status.** With `GEMINI_TIER=free`, the app stops at the first quota error and never continues or upgrades billing. But if the API key belongs to a project with billing linked, Google bills that usage at paid rates. To stay on the free tier, use a key from a project with **no** billing account linked.

**Stored data.** The app sends `store=false`, so interactions are not kept for later retrieval through the Interactions API. Google's [Interactions API overview](https://ai.google.dev/gemini-api/docs/interactions-overview) describes retention for stored interactions (1 day free, 55 days paid by default). Other logging and data use is governed by Google's terms.

### Illustrative cost example (not a quote, not a cap)

Rates for `gemini-3.1-flash-lite`, standard paid tier, from Google's pricing page on 2026-11-13: **$0.25 per 1M input tokens** and **$1.50 per 1M output tokens**. Thinking tokens are billed as output.

For one model call using 20,000 input tokens and 4,000 output tokens:

- input: 20,000 × $0.25 / 1,000,000 = **$0.005**
- output: 4,000 × $1.50 / 1,000,000 = **$0.006**
- total: about **$0.011** per call

The workflow makes 2–4 calls per question, and each call re-sends evidence, so a question costs a multiple of one call. With the default limits (40,000 characters in and 1,024 output tokens per call), a typical question uses far less than this example, but prices change: use the current pricing page. Google Search grounding is not used, so there are no search-grounding fees.

## Set up Tavily (web search)

1. Create an account at the [Tavily Platform](https://app.tavily.com/home) and copy your API key. You need a key to use web research.
2. Put it in `.env` as `TAVILY_API_KEY=tvly-...`, then restart the app. Never commit `.env`.
3. Check your usage in the Tavily dashboard.

**Tavily cost**, from Tavily's [API credits](https://docs.tavily.com/documentation/api-credits) page on 2026-11-13:

- The free plan includes 1,000 credits per month.
- A basic search costs 1 credit. This app only uses basic search, so at most `MAX_SEARCHES_PER_QUESTION` credits per question.
- Pay-as-you-go is $0.008 per credit beyond your plan.
- The app never turns on pay-as-you-go. Check your plan and billing settings at [Tavily pricing](https://www.tavily.com/pricing) and in your account.
- If Tavily reports a rate or credit limit, the app stops searching and tells you.

**Tavily credits are separate from Gemini token charges.** The app fetches pages itself, so Tavily's extract and crawl are never called. If no Tavily key is set, web research is turned off and Local documents mode with Ollama still works.

## POC boundaries

- Local documents: only `.txt`, `.md`, and text-based `.pdf` files, up to 5 MB each, in the configured folder only. Ollama only.
- Web access is read-only: GET requests only, no cookies kept, no sign-ins, form submissions, posting or email.
- `fetch_page` checks every URL:
  - only `http`/`https` on ports 80 and 443, with no user names or passwords in the URL
  - every hostname is resolved, and the fetch is blocked if any address is loopback, private, link-local (including cloud metadata), carrier-grade NAT, multicast or reserved
  - `localhost`, `.local` and similar names are blocked
  - the check is repeated on every redirect (at most 3), and the connection is pinned to the checked address
  - environment proxy settings are ignored for page fetches
- Fetched pages must be HTML or plain text. They are capped at 2 MB with a 10-second timeout, and extracted text is capped at 12,000 characters.
- Web text is wrapped as untrusted content, and the model is told never to follow instructions inside it. This reduces, but cannot remove, prompt-injection risk.
- No shell, database, write, credential or general browsing tools are available to the model.
- There are no accounts, persistent memory, background tasks or cloud hosting.
- The local web UI is for your own Mac. Don't expose its port to your network.

## Tests

Offline tests need no network, Ollama, or API keys:

```bash
python -m unittest discover -s tests
```

They cover:

- URL blocking, redirects, size limits and untrusted-content markers
- per-question call, search and fetch limits; spend-cap blocking; quota errors stopping without retry
- that local document text never reaches the research model
- the Gemini request settings (`store=false`, no continuation, output cap)
- that API keys never appear in errors

## Suggested first evaluation

Try ten questions:

- four with direct answers
- three that need several sources
- three whose answers are absent

Check that the assistant names the right sources and says when it cannot find support. Compare runs with the reviewer step on and off.

Also try one page containing instructions, for example a page telling the assistant to ignore its rules. Check that answers cite title, URL and date, and that the assistant does not act on the page's instructions.
