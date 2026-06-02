# NLP Fine-Tuning Dashboard

Interactive Streamlit dashboard visualising results from a supervised + preference fine-tuning project on **TinyLlama-1.1B**.

## What's Inside

| Section | Content |
|---|---|
| 🏠 Overview | Summary metrics, BLEU radar/bar chart, key findings |
| 📉 SFT Training | Loss curves for Trial 1 (LoRA r=8) and Trial 2 (qLoRA r=32) |
| 🎯 DPO Training | Reward curves for Trial 1 (β=0.1) and Trial 2 (β=0.5) |
| 📊 BLEU Evaluation | Per-prompt heatmap, grouped bar chart, full table |
| 🗣️ Manual Evaluation | Helpfulness / Harmlessness / Relevance radar & line charts |
| 🔬 Response Explorer | Side-by-side model response comparison for all 10 prompts |
| ⚙️ Model Configs | Full configuration table + pipeline architecture |

## Models Compared
- Base: TinyLlama-1.1B
- SFT Trial 1: LoRA r=8, q/v projections, 1 epoch
- SFT Trial 2: qLoRA r=32, all 7 projections, 2 epochs ← Best SFT
- DPO Trial 1: β=0.1, r=16, 1 epoch
- DPO Trial 2: β=0.5, r=32, 2 epochs ← Best DPO

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
See deployment instructions below.
