# Unified Dermatology Pipeline

This project provides a single, safety-first inference contract for dermatology VQA/classification.

## Quick Start

### 1) Install dependencies

```powershell
pip install -r requirements.txt
```

### 2) Run CLI inference

```powershell
python scripts/run_inference.py --image path/to/image.jpg --output 5_Results/inference_output.json
```

### 3) Run Streamlit app

```powershell
streamlit run app_streamlit.py
```

### 4) Smoke test (no heavy model load)

```powershell
python scripts/smoke_test_pipeline.py
```

### 5) Danh gia tap Data_test

```powershell
python scripts/evaluate_dataset.py --data_dir D:\DoAn_DaLieu\1_Data\Data_test
```

### 6) Ve bieu do hoc thuat

```powershell
python scripts/plot_metrics.py
```

### 7) LLM post-processor (tuy chon)

Dat bien moi truong OPENAI_API_KEY de bat LLM trong UI.

### 8) Sanity check rieng cho classification model

Chay unit test doc lap cho model classification, khong qua ROI, segmentation, safety gate hay Streamlit:

```powershell
python scripts/sanity_check_cls.py --image path/to/image.jpg
```

Neu muon chi ro weights:

```powershell
python scripts/sanity_check_cls.py --image path/to/image.jpg --weights 4_Models/classification/efficientnet_attention_best.pth
```

Script se in ra xac suat cua toan bo 7 lop, nhan du doan cao nhat, va su dung bang nhan ISIC co dinh.

## Notes
- Models are loaded once via a singleton registry to reduce CPU overhead.
- The safety gate rejects predictions with low confidence or poor segmentation quality.
