## Limitations
- The auditor checks **category** consistency only, not brand or attributes. Tested example:
  a Seiko watch photo with the title "Fastrack Men Analog Watch" and category Watches is marked OK.
  Brand verification (logo recognition, OCR on the product, or image–text matching models like CLIP)
  would be needed to catch this.

## FINDINGS
- ~2/3 of the image model's 127 disagreements come from two look-alike pairs:
  Shoes ↔ Sandal (61) and Topwear ↔ Innerwear (25). The asymmetry (40 shoes predicted as
  sandals vs 21 the other way) is consistent with sandals being filed under Shoes in the catalog.
## Data & attribution
Product images and metadata: *Fashion Product Images (Small)* by Param Aggarwal on Kaggle
(released under the MIT License). Product photos originally from Myntra.
This project is a non-commercial portfolio demonstration.
# 🛍️ Catalog Quality Auditor

**Live app:** https://catalog-auditor-nikhil.streamlit.app

A multimodal ML system that flags **mislabeled e-commerce listings** — products whose seller-chosen category doesn't match what's actually in the photo and title.

On marketplaces with thousands of small sellers, listings often get filed under the wrong category (sunglasses under *Topwear*, sandals under *Shoes*). Wrong categories break search, browsing and recommendations. This project builds an automated auditor that catches those errors and routes doubtful cases to human review.

---

## How it works

Every listing gets **three independent opinions**:

| Opinion | Source |
|---|---|
| Seller's label | The category the seller chose |
| Text model | Predicts the category from the **product title** (TF-IDF + Logistic Regression) |
| Image model | Predicts the category from the **product photo** (MobileNetV2, transfer learning) |

The auditor compares them:

| Situation | Verdict |
|---|---|
| All three agree | ✅ **OK** |
| Text and image agree with each other, but both disagree with the seller | ⚠️ **Likely mislabel** — auto-fix candidate |
| Only one model agrees with the seller | 🔍 **Needs review** |
| All three disagree | 🔍 **Needs review** |

Two independent models agreeing *against* the label is much stronger evidence than either model alone — it's unlikely both make the same mistake.

---

## Results

**Dataset:** [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) — 44,419 products with photos and titles. Kept the 10 largest categories → **36,958 products** (29,566 train / 7,392 test, stratified, one shared split for both models).

### Model accuracy (7,392 held-out test listings)
| Model | Accuracy |
|---|---|
| Text model (TF-IDF + Logistic Regression) | **99.6%** |
| Image model (MobileNetV2, fine-tuned) | **98.3%** |

### Mislabel detection — injected-error experiment
To measure detection with known ground truth, **5% of test labels (369 listings) were deliberately corrupted**, then audited:

| Error type | Recall (Likely mislabel) | Precision | Recall (any flag) |
|---|---|---|---|
| Random wrong category | **98.1%** | **98.6%** | 100% |
| Realistic look-alike (e.g. shoes ↔ sandal) | **97.8%** | **98.6%** | 99.7% |

Precision is conservative: genuine catalog mislabels that weren't part of the injection count as false alarms.

### Real catalog errors found
On the untouched test set, **98.0%** of listings passed, and the auditor surfaced real problems, e.g.:
- **Sunglasses filed as Topwear / Tunics** — both models say Eyewear (image model 100% confident)
- **Sandals and floaters filed under Shoes** — the catalog files sandals inconsistently

### Where errors concentrate
- **Sandal** is the most-flagged category (~12% of its listings), driven by sandals being filed under *Shoes*.
- About **2/3 of the image model's 127 disagreements** come from two look-alike pairs: **Shoes ↔ Sandal (61)** and **Topwear ↔ Innerwear (25)**.
- The two models cover each other's blind spots: vague titles like *"Levis Kids Boy's Aldon White Kidswear"* never name the product type, but the photo clearly shows shorts — **Bottomwear is the one category where the image model beats the text model**.

---

## The app

| Page | What it does |
|---|---|
| **Check a listing** | Upload a photo (or pick a catalog sample), enter a title and category → both models' predictions, confidence, and a verdict |
| **Bulk catalog audit** | Audit results for 7,392 listings, filterable by verdict and category, with photos and CSV export |
| **Insights** | Model accuracy, flag rate by category, text-vs-image accuracy, confusion heatmap, top disagreements |

---

## Approach

1. **EDA** — category imbalance (Topwear ~15,400 vs Sandal ~960), 60×80 images (~1% grayscale → converted to RGB), titles average ~5.7 words and usually name the product type; clothing photos often show full outfits on models.
2. **Text model** — TF-IDF (unigrams + bigrams) + Logistic Regression with balanced class weights.
3. **Image model** — MobileNetV2 pretrained on ImageNet at 96×96: trained a new classification head on a frozen base, then fine-tuned the top 30 layers (BatchNorm frozen, learning rate 1e-5), with augmentation and class weights.
4. **Audit logic** — compare seller label, text prediction and image prediction into four verdict buckets.
5. **Evaluation** — injected-mislabel experiment (random and look-alike errors) plus manual review of real flags.
6. **Deployment** — Streamlit app on Streamlit Community Cloud.

**Tech:** Python, pandas, scikit-learn, TensorFlow/Keras, Streamlit, Altair, Kaggle (GPU training).

---

## Limitations

- **Category only, not brand or attributes.** Tested example: a Seiko watch photo with the title *"Fastrack Men Analog Watch"* and category Watches is marked OK. Catching brand mismatches would need logo recognition, OCR, or image–text matching (e.g. CLIP).
- **Systematic labeling conventions are harder to catch than one-off mistakes.** When many sellers consistently file sandals under Shoes, the models learn that convention during training. These cases mostly land in *Needs review* rather than *Likely mislabel*.
- **Low-resolution, catalog-style images.** Trained on 60×80 photos with plain backgrounds; cluttered phone photos may be predicted less reliably.
- **Fixed set of 10 categories.** Products outside them are forced into the closest category.
- The image model sometimes predicts *Bags* for strappy items (sports bras, sandals). Because the text model disagrees, these go to human review instead of being auto-fixed.

## Future work
- Brand/attribute verification (OCR, CLIP-style image–text matching)
- Visual similarity search for duplicate listings
- Full-stack version (FastAPI backend + React frontend)

---

## Run locally

```bash
git clone https://github.com/nikhilreddynani1-beep/catalog-auditor.git
cd catalog-auditor
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.10–3.13 (TensorFlow 2.20).

## Repository structure
```
catalog-auditor/
├── app.py                 # Streamlit app (3 pages)
├── requirements.txt       # pinned versions matching the training environment
├── test_load.py           # sanity check that models load and predict
└── app_artifacts/
    ├── image_model.keras           # fine-tuned MobileNetV2
    ├── tfidf.joblib                # TF-IDF vectorizer
    ├── text_model.joblib           # Logistic Regression
    ├── class_names.json            # class order for the image model
    ├── predictions_with_audit.csv  # test-set predictions + verdicts
    └── images/                     # flagged products + sample thumbnails
```

## Data & attribution
Product images and metadata: *Fashion Product Images (Small)* by Param Aggarwal on Kaggle (released under the MIT License). Product photos originally from Myntra. This project is a non-commercial portfolio demonstration.
