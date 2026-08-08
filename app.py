import streamlit as st
import requests
from datetime import datetime


st.set_page_config(
    page_title="AI Caption Generator",
    layout="wide"
)

st.title("AI Viral Caption Generator")

# user input
prompt = st.text_area(
    "Describe Your vedio you want to generate caption with hashtags"
)

followers_value = st.number_input("Followers (numeric part)", 0.0)
followers_unit = st.selectbox("Followers Unit", options=["", "K", "M"])

posts = st.number_input("Total Posts", 0)

reel_duration = st.number_input(
    "Duration (sec)",
    min_value=1,
    value=20
)

date_input = st.date_input("Select Date")
time_str = st.text_input("Enter Time (HH:MM:SS, 24-hour format)", value="12:00:00")

# Validate time format
try:
    time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
    valid_time = True
except ValueError:
    st.error("Please enter time in HH:MM:SS (24-hour) format")
    valid_time = False

# Helper function to normalize 
def normalize_count(value: float, unit: str) -> int:
    unit = unit.upper()
    if unit == "K":
        return int(value * 1_000)
    elif unit == "M":
        return int(value * 1_000_000)
    else:
        return int(value)

# Generate Button

if st.button("Generate Captions"):
    # empty prompt check
    if not prompt.strip():

        st.warning(
            "Please enter video description."
        )

        st.stop()

    # invalid time check
    if not valid_time:
        st.stop()

    followers = normalize_count(followers_value, followers_unit)

    # Combine date + time into proper datetime string
    datetime_str = datetime.combine(date_input, time_obj).strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "description": prompt,
        "followers": followers,
        "total_posts": posts,
        "duration_sec": reel_duration,
        "datetime": datetime_str
    }

    # API request

    try:

        with st.spinner(
            "Generating viral captions..."
        ):

            response = requests.post(
                "http://localhost:8000/generate",
                json=payload,
                timeout=300
            )

            if response.status_code != 200:

                st.error(response.text)
                st.stop()

            data = response.json()

    except requests.exceptions.RequestException as e:
        st.error(str(e))

        st.stop()
    
    # Display result

    st.success("Captions Generated Successfully")

    best = data["best_caption"]

    # best Caption
    st.divider()

    st.success("Top Recommended Caption")

    st.markdown(
        f"""
### Engagement Potential: {best['engagement_score']}%

{best['caption']}

{best['hashtags']}
"""
    )

    # All caption

    st.divider()
    st.subheader("All Ranked Captions")

    for item in data["all_captions"]:

        with st.container(border=True):
            st.markdown(
                f"""
### Engagement Potential: {item['engagement_score']}

{item['caption']}

{item['hashtags']}
"""
            )
