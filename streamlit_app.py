import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="TikCheck App",
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


# Now final DataFrame with true label included
final = pd.read_csv("final_predicted_on_test.csv")
merged = df.merge(final, on="review_id", how="right")
# --- Streamlit UI ---
logo_path = Path(__file__).parent / "tiktok_logo.png"
st.image(logo_path, width=100)
import base64
from pathlib import Path

badge_path = Path(__file__).parent / "tikcheck.jpg"
with open(badge_path, "rb") as f:
    data = f.read()
encoded = base64.b64encode(data).decode()

st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:center;gap:1px;">
    <img src="data:image/png;base64,{encoded}" width="300">
</div>
""", unsafe_allow_html=True)


st.markdown(
    """
    ### Introduction
    Online reviews play a crucial role in shaping public perception of local businesses.  
    This app helps **assess review quality & relevancy** using ML/NLP methods.  
    Explore reviews for different businesses and see our predicted labels.
    """
)
''

# Business dropdown
business_list = merged["business_name"].dropna().unique()
selected_business = st.selectbox("🏢 Select a business to explore:", sorted(business_list))

# Filter reviews for the chosen business
filtered_df = merged[merged["business_name"] == selected_business]
''

st.subheader(f"Reviews for: **{selected_business}**")
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


# Display reviews + predictions

''

for _, row in filtered_df.iterrows():
    # Set background color based on label
    if row['final_label'] == "relevant":
        bg_color = "#d4f7dc"  # light green
    else:
        bg_color = "#f1f3f6"  # default gray

    st.markdown(
        f"""
        <div style="
            border-radius: 15px;
            padding: 12px 18px;
            margin: 8px 0;
            background-color: {bg_color};
            box-shadow: 0px 1px 3px rgba(0,0,0,0.1);
            max-width: 700px;">
            <b>⭐ {row['rating']}</b> - This review is classified as <i>{row['final_label']}</i><br>
            {row['review']}
        </div>
        """,
        unsafe_allow_html=True,
    )

