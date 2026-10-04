import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import keras
from PIL import Image

ART = 'app_artifacts'
IMG_DIR = os.path.join(ART, 'images')

st.set_page_config(page_title="Catalog Quality Auditor", page_icon="🛍️", layout="wide")

# ---------- Load models and data once (cached, so the app stays fast) ----------
@st.cache_resource
def load_models():
    image_model = keras.models.load_model(os.path.join(ART, 'image_model.keras'))
    tfidf = joblib.load(os.path.join(ART, 'tfidf.joblib'))
    text_model = joblib.load(os.path.join(ART, 'text_model.joblib'))
    with open(os.path.join(ART, 'class_names.json')) as f:
        class_names = json.load(f)
    return image_model, tfidf, text_model, class_names

@st.cache_data
def load_catalog():
    df = pd.read_csv(os.path.join(ART, 'predictions_with_audit.csv'))
    available = set(os.listdir(IMG_DIR))
    df = df[df['image_path'].isin(available)]
    # Show the most interesting products first in the sample picker
    priority = {'LIKELY MISLABEL': 0, 'ALL DIFFERENT': 1, 'ONE MODEL DISAGREES': 2, 'OK': 3}
    df = df.assign(_p=df['audit'].map(priority)).sort_values('_p').drop(columns='_p')
    return df.reset_index(drop=True)

@st.cache_data
def load_full_results():
    return pd.read_csv(os.path.join(ART, 'predictions_with_audit.csv'))
image_model, tfidf, text_model, class_names = load_models()
catalog = load_catalog()
full = load_full_results()

# ---------- Prediction + audit logic (same as the notebook) ----------
def predict_image(img):
    img = img.convert('RGB').resize((96, 96), Image.BILINEAR)
    x = np.array(img, dtype='float32')[None, ...]
    probs = image_model.predict(x, verbose=0)[0]
    return class_names[int(probs.argmax())], float(probs.max())

def predict_text(title):
    probs = text_model.predict_proba(tfidf.transform([title]))[0]
    idx = int(probs.argmax())
    return text_model.classes_[idx], float(probs[idx])

def audit(label, text_pred, img_pred):
    if text_pred == label and img_pred == label:
        return 'OK'
    if text_pred == img_pred and text_pred != label:
        return 'LIKELY MISLABEL'
    if text_pred == label or img_pred == label:
        return 'ONE MODEL DISAGREES'
    return 'ALL DIFFERENT'

# ---------- Sidebar navigation ----------
st.sidebar.title("🛍️ Catalog Auditor")
page = st.sidebar.radio("Go to", ["Check a listing", "Bulk catalog audit", "Insights"])
st.sidebar.caption("Flags e-commerce listings whose category doesn't match their photo and title.")

# ---------- Page 1: Check a listing ----------
if page == "Check a listing":
    st.title("Check a listing")
    st.write("The auditor compares the seller's chosen category with what the **image model** "
             "sees in the photo and what the **text model** reads in the title.")

    source = st.radio("Product source", ["Try a sample from the catalog", "Upload my own"], horizontal=True)

    if source == "Try a sample from the catalog":
        options = [f"{r.id} — {r.productDisplayName}" for r in catalog.itertuples()]
        choice = st.selectbox("Pick a product (flagged ones are listed first)", options)
        row = catalog.iloc[options.index(choice)]
        image = Image.open(os.path.join(IMG_DIR, row['image_path']))
        title = st.text_input("Product title", row['productDisplayName'])
        label = st.selectbox("Seller's category", class_names, index=class_names.index(row['subCategory']))
    else:
        uploaded = st.file_uploader("Product photo", type=['jpg', 'jpeg', 'png'])
        image = Image.open(uploaded) if uploaded else None
        title = st.text_input("Product title", placeholder="e.g. Puma Men Grey T-shirt")
        label = st.selectbox("Seller's category", class_names)
        st.caption("Works best with catalog-style photos: a single product on a plain background.")

    if st.button("Run audit", type="primary"):
        if image is None or not title.strip():
            st.error("Please provide both a photo and a title.")
        else:
            img_pred, img_conf = predict_image(image)
            text_pred, text_conf = predict_text(title)
            verdict = audit(label, text_pred, img_pred)

            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(image, caption="Product photo", width=180)
            with col2:
                c1, c2, c3 = st.columns(3)
                c1.metric("Seller's label", label)
                c2.metric("Text model", text_pred)
                c2.caption(f"{text_conf:.0%} confident")
                c3.metric("Image model", img_pred)
                c3.caption(f"{img_conf:.0%} confident")

                if verdict == 'OK':
                    st.success("✅ **OK** — the label matches both the photo and the title.")
                elif verdict == 'LIKELY MISLABEL':
                    st.error(f"⚠️ **Likely mislabel** — both models independently say **{text_pred}**, "
                             f"not **{label}**. Suggest correcting the category.")
                elif verdict == 'ONE MODEL DISAGREES':
                    st.warning("🔍 **Needs review** — one model agrees with the label, the other doesn't.")
                else:
                    st.warning("🔍 **Needs review** — the label, title and photo all point to different categories.")

# ---------- Pages 2 and 3 (coming next) ----------
elif page == "Bulk catalog audit":
    st.title("Bulk catalog audit")
    st.write(f"Audit results for **{len(full):,}** held-out catalog listings, checked by both models.")

    counts = full['audit'].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total listings", f"{len(full):,}")
    c2.metric("✅ OK", f"{counts.get('OK', 0):,}")
    c3.metric("🔍 Needs review", f"{counts.get('ONE MODEL DISAGREES', 0) + counts.get('ALL DIFFERENT', 0):,}")
    c4.metric("⚠️ Likely mislabel", f"{counts.get('LIKELY MISLABEL', 0):,}")

    st.divider()
    f1, f2 = st.columns(2)
    verdicts = f1.multiselect("Verdict", ['LIKELY MISLABEL', 'ALL DIFFERENT', 'ONE MODEL DISAGREES', 'OK'],
                              default=['LIKELY MISLABEL', 'ALL DIFFERENT'])
    categories = f2.multiselect("Seller's category", class_names, default=class_names)

    view = full[full['audit'].isin(verdicts) & full['subCategory'].isin(categories)]
    st.write(f"Showing **{len(view):,}** listings")

    # Photo gallery (only for products whose image was bundled with the app)
    available = set(os.listdir(IMG_DIR))
    gallery = view[view['image_path'].isin(available)].head(24)
    if len(gallery):
        cols = st.columns(6)
        for i, r in enumerate(gallery.itertuples()):
            with cols[i % 6]:
                st.image(os.path.join(IMG_DIR, r.image_path), width=110)
                st.caption(f"**Label:** {r.subCategory}  \n**Text:** {r.text_pred}  \n**Image:** {r.img_pred}")

    st.dataframe(
        view[['id', 'productDisplayName', 'subCategory', 'text_pred', 'img_pred', 'img_conf', 'audit']]
            .rename(columns={'productDisplayName': 'title', 'subCategory': 'seller_label',
                             'img_conf': 'image_confidence'}),
        use_container_width=True, hide_index=True
    )

    st.download_button("⬇️ Download these listings as CSV",
                       view.to_csv(index=False).encode('utf-8'),
                       file_name='flagged_listings.csv', mime='text/csv')
else:
    import altair as alt

    st.title("Insights")

    # ---- Headline numbers ----
    st.subheader("Model performance")
    text_acc = (full['text_pred'] == full['subCategory']).mean()
    img_acc = (full['img_pred'] == full['subCategory']).mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Text model accuracy", f"{text_acc:.1%}")
    c2.metric("Image model accuracy", f"{img_acc:.1%}")
    c3.metric("Injected-error recall", "97.8–98.1%")
    c4.metric("Flag precision", "98.6%")
    st.caption("Recall and precision come from an offline experiment where 5% of test labels "
               "(369 listings) were deliberately corrupted.")

    # ---- Where flags happen ----
    st.subheader("Which categories get flagged most?")
    flag_df = (full.assign(flagged=full['audit'] != 'OK')
                   .groupby('subCategory')['flagged'].mean()
                   .mul(100).rename('Flagged %').reset_index())
    flag_chart = alt.Chart(flag_df).mark_bar().encode(
        x=alt.X('Flagged %:Q', title='% of listings flagged'),
        y=alt.Y('subCategory:N', sort='-x', title=None),
        tooltip=['subCategory', alt.Tooltip('Flagged %:Q', format='.1f')])
    st.altair_chart(flag_chart, use_container_width=True)
    st.caption("Share of each category's listings that were not marked OK.")

    # ---- Per-category accuracy, text vs image ----
    st.subheader("Per-category accuracy: text vs image")
    per_cat = (full.assign(**{'Text model': full['text_pred'] == full['subCategory'],
                              'Image model': full['img_pred'] == full['subCategory']})
                   .groupby('subCategory')[['Text model', 'Image model']].mean().mul(100))
    acc_df = per_cat.reset_index().melt(id_vars='subCategory', var_name='Model', value_name='Accuracy %')
    acc_chart = alt.Chart(acc_df).mark_bar(clip=True).encode(
        x=alt.X('Accuracy %:Q', scale=alt.Scale(domain=[80, 100]), title='Accuracy (%)'),
        y=alt.Y('subCategory:N', title=None),
        yOffset='Model:N',
        color=alt.Color('Model:N', legend=alt.Legend(orient='bottom')),
        tooltip=['subCategory', 'Model', alt.Tooltip('Accuracy %:Q', format='.1f')])
    st.altair_chart(acc_chart, use_container_width=True)

    # ---- Image model confusion heatmap ----
    st.subheader("Image model: what gets confused with what")
    cm = (pd.crosstab(full['subCategory'], full['img_pred'])
            .reset_index()
            .melt(id_vars='subCategory', var_name='Predicted', value_name='Count'))
    base = alt.Chart(cm).encode(
        x=alt.X('Predicted:N', title='Image model prediction'),
        y=alt.Y('subCategory:N', title="Seller's label"))
    heat = base.mark_rect().encode(
        color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues', type='symlog'), legend=None),
        tooltip=['subCategory', 'Predicted', 'Count'])
    labels = base.mark_text(fontSize=11).encode(
        text='Count:Q',
        color=alt.condition(alt.datum.Count > 300, alt.value('white'), alt.value('black'))
    ).transform_filter(alt.datum.Count > 0)
    st.altair_chart(heat + labels, use_container_width=True)

    # ---- Top confusions table ----
    st.subheader("Most common disagreements")
    top_conf = (full[full['img_pred'] != full['subCategory']]
                .groupby(['subCategory', 'img_pred']).size()
                .sort_values(ascending=False).head(8)
                .reset_index(name='Listings')
                .rename(columns={'subCategory': "Seller's label", 'img_pred': 'Image model says'}))
    st.dataframe(top_conf, hide_index=True, use_container_width=True)
    st.caption("Some of these are model mistakes; others reflect inconsistent catalog labels, "
               "e.g. sandals filed under Shoes.")