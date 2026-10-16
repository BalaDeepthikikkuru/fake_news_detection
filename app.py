import streamlit as st
import joblib

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

model = joblib.load("fake_news_model.pkl")

st.title("📰 Fake News Detection System")

st.markdown(
    """
    ### Welcome!
    Enter a news article below to get a prediction
    from our NLP and Machine Learning model.
    """
)

st.info(
    "How to use: Paste a news article, then click Check News."
)

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

def clear_form():
    st.session_state.news_text = ""

news = st.text_area(
    "✍️ Paste your news article here",
    height=200,
    key="news_text",
    placeholder="Paste the full news article here..."
)

col1, col2 = st.columns(2)

with col1:
    check = st.button("🔎 Check News", type="primary",
                      use_container_width=True)

with col2:
    st.button("🧹 Clear", on_click=clear_form,
              use_container_width=True)

if check:
    if len(news.strip()) < 50:
        st.warning("Please enter at least 50 characters.")
    else:
        prediction = model.predict([news])[0]
        probabilities = model.predict_proba([news])[0]
        confidence = max(probabilities) * 100

        st.subheader("📊 Analysis Result")

        if prediction == 0:
            st.error("Prediction: Likely Fake")
        else:
            st.success("Prediction: Likely Genuine")

        st.metric("Model confidence", f"{confidence:.2f}%")

        st.caption(
            "Confidence reflects the model's output, not proof "
            "that the news is true or false. Verify claims using "
            "reliable sources."
        )

st.divider()
st.caption("Fake News Detection | NLP + Machine Learning | Python")