import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Review Quality & Relevancy App",
    layout="wide",
    page_icon="📝"
)

@st.cache_data
def get_data():
    """Grab main review data from CSV file with caching."""
    DATA_FILENAME = Path(__file__).parent / 'cleaned_dataset.csv'
    df = pd.read_csv(DATA_FILENAME)
    return df

df = get_data()

# --- Load model outputs ---
rule_based_df = pd.read_csv('rule_based.csv')
dnn_df = pd.read_csv('dnn.csv')
import pandas as pd

# Make sure both DataFrames have consistent column names
# Suppose the original label columns are 'label' and predicted columns are 'pred_rule' / 'pred_dnn'

# Rename columns in rule-based df
rule_based_df = rule_based_df.rename(columns={
    'pred_rule': 'pred_rule'    # predicted by rule-based model
})

# Rename columns in DNN df
dnn_df = dnn_df.rename(columns={
    'pred_dnn': 'pred_dnn'      # predicted by DNN
})

# Merge on review_id
merged = pd.merge(rule_based_df, dnn_df, on='review_id', suffixes=('_rule', '_dnn'))

# Columns for class probabilities
classes = ["ads", "irrelevant", "rant", "relevant", "spam"]

# Compute weighted final scores
w_rule = 0.4
w_dnn = 0.6
for cls in classes:
    merged[f"{cls}_final"] = w_rule * merged[f"{cls}_rule"] + w_dnn * merged[f"{cls}_dnn"]

# Case 1: If predictions agree → keep that label
merged["final_label"] = merged.apply(
    lambda row: row["pred_rule"] if row["pred_rule"] == row["pred_dnn"] else None, axis=1
)

# Case 2: If predictions differ → pick max weighted score
mask_disagree = merged["final_label"].isna()
merged.loc[mask_disagree, "final_label"] = merged.loc[mask_disagree, [f"{cls}_final" for cls in classes]].idxmax(axis=1)
merged["final_label"] = merged["final_label"].str.replace("_final", "")

# Now final DataFrame with true label included
final = merged[["review_id", "pred_rule", "pred_dnn", "final_label", "true_label_rule"]]
# --- Streamlit UI ---
st.title("📝 Google Location Reviews Quality Checker")
st.markdown(
    """
    ### Introduction
    Online reviews play a crucial role in shaping public perception of local businesses.  
    This app helps **assess review quality & relevancy** using ML/NLP methods.  
    Explore reviews for different businesses and see predicted labels.
    """
)
''

# Business dropdown
business_list = df["business_name"].dropna().unique()
selected_business = st.selectbox("🏢 Select a business to explore:", sorted(business_list))

# Filter reviews for the chosen business
filtered_df = df[df["business_name"] == selected_business]
''

st.subheader(f"📊 Reviews for: **{selected_business}**")
st.markdown(f"**Total Reviews:** {len(filtered_df)}")

# Show business description
if "business_desc" in df.columns:
    # Get the first non-NaN description
    desc_list = filtered_df["business_desc"].iloc[0] if not filtered_df["business_desc"].isna().all() else []

    # If it's a string representation of a list, convert to actual list
    if isinstance(desc_list, str):
        import ast
        try:
            desc_list = ast.literal_eval(desc_list)
        except:
            desc_list = [desc_list]

    # Join with commas
    desc_text = ", ".join(desc_list) if desc_list else "N/A"

    # Display bolded
    st.markdown(f"**Description:** {desc_text}")

# ✅ Merge with final predictions on review_id
filtered_with_preds = filtered_df.merge(final, on="review_id", how="inner")

# Display reviews + predictions

''

for _, row in filtered_with_preds.iterrows():
    st.markdown(
        f"""
        <div style="
            border-radius: 15px;
            padding: 12px 18px;
            margin: 8px 0;
            background-color: #f1f3f6;
            box-shadow: 0px 1px 3px rgba(0,0,0,0.1);
            max-width: 700px;">
            <b>⭐ {row['rating']}</b> - This review is classified as <i>{row['final_label']}</i><br>
            {row['review']}
        </div>
        """,
        unsafe_allow_html=True,
    )

