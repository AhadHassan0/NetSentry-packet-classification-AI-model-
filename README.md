# NetSentry-packet-classification-AI-model-
NetSentry is an AI model built to classify data packets into Benign or Malicious

# 🛡️ NetSentry

AI-powered network intrusion detection. Classifies CICIDS2017 CSV or raw `.pcap`
traffic as **benign or malicious** and surfaces high-risk source IPs on a live dashboard.

---

## Project Structure

```
NetSentry/
├── preprocess_data.py   # Step 1 – clean raw CICIDS2017 CSV
├── train_model.py       # Step 2 – train XGBoost model
├── app.py               # Step 3 – Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## How to Run

### 0 — Prerequisites

- Python 3.10 or 3.11
- (Optional) tshark for `.pcap` support → install [Wireshark](https://www.wireshark.org/download.html)

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Add your raw data

Download any CICIDS2017 CSV from https://www.unb.ca/cic/datasets/ids-2017.html
and rename it (or update `INPUT_CSV` in `preprocess_data.py`):

```
NetSentry/
└── raw_traffic.csv    ← place it here
```

### 3 — Preprocess

```bash
python preprocess_data.py
```

Produces `cleaned_training_data.csv`.

### 4 — Train

```bash
python train_model.py
```

Produces `netsentry_model.pkl`. Training on CPU with 500k rows takes ~5–10 min.
If a CUDA GPU is detected it switches automatically.

### 5 — Launch the dashboard

```bash
streamlit run app.py
```

Opens at http://localhost:8501

---

## Using the Dashboard

| Upload type | What you get |
|---|---|
| CICIDS2017 `.csv` | Full classification + high-risk IPs (if Source IP column present) |
| Raw `.pcap` | tshark extracts flow features → classification + high-risk IPs |

1. Upload your file using the sidebar.
2. Review the **Traffic Summary** metrics and charts.
3. Check **High-Risk Source IPs** — sorted by malicious hit count.
4. Inspect or download **Flagged Malicious Records**.

---

## Notes

- The model is trained for binary classification: **Benign vs Malicious**.
- `.pcap` support requires `tshark`. Without it, use a pre-extracted CSV.
- NetSentry is designed to complement (not replace) traditional firewalls/IDS tools.
