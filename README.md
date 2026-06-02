# TinyLlama Fine-Tuning Lab — Interactive Dashboard

An interactive Streamlit dashboard built as the **Application deliverable** for Assignment 4 (NLP with Deep Learning). It visualises all results from supervised (SFT) and preference (DPO) fine-tuning of **TinyLlama-1.1B** on Kaggle.

---

## Pages

| Page | What it does |
|---|---|
| 🏠 Overview | BLEU metric cards, radar chart, bar chart, key findings |
| 🎮 Model Playground | **Pick any of the 10 eval prompts → choose models → see actual responses side-by-side with BLEU scores and H/H/R bars** |
| 📉 Training Curves | Interactive loss + reward curves for all 4 fine-tuned models across tabs |
| 📊 BLEU Analysis | Filterable grouped bar, heatmap, delta-vs-base chart, full table |
| 🗣️ Quality Evaluator | Slide through prompts, star-rated H/H/R scores per model, radar + line charts |
| 🔬 Config Explorer | **Sliders for LoRA rank / modules / epochs and DPO β — shows estimated BLEU, reward margin, drift risk live** |

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy on Streamlit Cloud (step-by-step)

### Step 1 — Create a GitHub repo
1. Go to https://github.com/new
2. Name it e.g. `tinyllama-finetuning-lab`, set to **Public**
3. Click **Create repository**

### Step 2 — Upload these files
Upload all four files to the repo root:
```
app.py
requirements.txt
README.md
```
For the config file, click **Add file → Create new file**, type the path as:
```
.streamlit/config.toml
```
and paste the contents of `config.toml`.

### Step 3 — Deploy
1. Go to https://share.streamlit.io — sign in with GitHub
2. Click **New app**
3. Select your repo → branch `main` → main file `app.py`
4. Click **Deploy** — live in ~2 minutes

---

## Models Compared

| Model | Method | Rank | Modules | Avg BLEU | Manual Score |
|---|---|---|---|---|---|
| Base | — | — | — | 1.34 | — |
| SFT Trial 1 | LoRA | 8 | q, v | 1.54 | — |
| SFT Trial 2 ★ | qLoRA | 32 | All 7 | 1.62 | 4.23/5 |
| DPO Trial 1 | DPO + LoRA | 16 | q,k,v,o | 1.28 | 4.40/5 |
| DPO Trial 2 ★ | DPO + qLoRA | 32 | All 7 | 1.42 | 4.90/5 |

---

## Datasets
- **SFT**: `databricks/databricks-dolly-15k` — 5,000 samples, Alpaca prompt format
- **DPO**: `Intel/orca_dpo_pairs` — 2,000 samples, prompt/chosen/rejected format
- **Eval**: 10 out-of-distribution prompts with ChatGPT-4o reference answers

## Platform
Kaggle Notebook · Tesla T4 GPU · 15.6 GB VRAM
