import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
import base64
import plotly.express as px
from wordcloud import WordCloud
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# --- INITIALIZATION ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

st.set_page_config(page_title="Brand Reputation Dashboard", layout="wide")

# --- LOCAL BACKGROUND IMAGE & AESTHETIC CSS ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg_local(main_bg_img):
    try:
        bin_str = get_base64_of_bin_file(main_bg_img)
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{bin_str}");
                background-size: cover;
                background-attachment: fixed;
            }}
            .block-container {{
                background: rgba(255, 255, 255, 0.90);
                backdrop-filter: blur(10px);
                padding: 3rem;
                border-radius: 25px;
                margin-top: 2rem;
                border: 1px solid rgba(255, 255, 255, 0.3);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
            }}
            .main-title {{
                text-align: center; 
                color: #000000; 
                font-family: 'Inter', sans-serif;
                font-weight: 800;
                letter-spacing: -1px;
                margin-bottom: 20px;
            }}
            .priority-card {{ 
                background: white; 
                border-left: 6px solid #f43f5e; 
                padding: 20px; 
                margin-bottom: 15px; 
                border-radius: 12px;
                box-shadow: 2px 4px 12px rgba(0,0,0,0.05);
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        pass

set_bg_local('bg.png') 

# --- SESSION STATE ---
if "page_view" not in st.session_state: st.session_state.page_view = "upload"
if "data" not in st.session_state: st.session_state.data = None
if "text_col" not in st.session_state: st.session_state.text_col = None
if "plat_col" not in st.session_state: st.session_state.plat_col = None

# --- CORE FUNCTIONS ---
def clean_text(text):
    if pd.isna(text) or not isinstance(text, str): return ""
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    return " ".join([stemmer.stem(w) for w in tokens if w not in stop_words])

@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    model = load_model(os.path.join(base_path, 'lstm_sentiment_model.keras'))
    with open(os.path.join(base_path, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()

def process_sentiment(df, text_col):
    df['Cleaned'] = df[text_col].apply(clean_text)
    seqs = tokenizer.texts_to_sequences(df['Cleaned'])
    padded = pad_sequences(seqs, maxlen=100)
    preds = model.predict(padded)
    df['Sentiment'] = np.argmax(preds, axis=1)
    df['Sentiment'] = df['Sentiment'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})
    return df

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Main Menu")
    selection = st.radio("Navigate to:", [
        "📂 Dataset Dashboard", 
        "✍️ Manual Analysis", 
        "🚨 Review Analysis", 
        "⚖️ Product Comparison", 
        "📊 Product Sentiment", 
        "🔥 Brand Crisis Detection"
    ])
    st.markdown("---")
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

# --- AESTHETIC COLORS ---
AESTHETIC_COLORS = {'Positive':'#10B981', 'Neutral':'#3B82F6', 'Negative':'#F43F5E'}

# --- PAGE ROUTING ---

if selection == "📂 Dataset Dashboard":
    st.markdown("<h1 class='main-title'>🛡️ Brand Monitoring Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.page_view == "upload":
        st.subheader("Upload Global Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            df_raw = pd.read_csv(uploaded_file, encoding='latin1')
            df_raw.columns = df_raw.columns.str.strip() 
            c1, c2 = st.columns(2)
            t_col = c1.selectbox("Select Review Text Column", df_raw.columns)
            p_col = c2.selectbox("Select Platform/Source Column", df_raw.columns)
            if st.button("Generate Analytics", type="primary"):
                with st.spinner("Analyzing Sentiments..."):
                    df = df_raw.dropna(subset=[t_col, p_col]).copy()
                    st.session_state.data = process_sentiment(df, t_col)
                    st.session_state.text_col, st.session_state.plat_col = t_col, p_col
                    st.session_state.page_view = "dashboard"
                    st.rerun()

    elif st.session_state.page_view == "dashboard":
        df = st.session_state.data
        counts = df['Sentiment'].value_counts()
        total = len(df)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Posts", total)
        m2.metric("Positive", counts.get('Positive',0), f"{(counts.get('Positive',0)/total)*100:.1f}%")
        m3.metric("Neutral", counts.get('Neutral',0), f"{(counts.get('Neutral',0)/total)*100:.1f}%")
        m4.metric("Negative", counts.get('Negative',0), f"{(counts.get('Negative',0)/total)*100:.1f}%")

        c_vol, c_plat = st.columns([1.5, 3])
        with c_vol:
            st.subheader("🌐 Source Volume")
            v_data = df[st.session_state.plat_col].value_counts().reset_index()
            st.plotly_chart(px.bar(v_data, y=st.session_state.plat_col, x='count', orientation='h', color=st.session_state.plat_col), use_container_width=True)

        with c_plat:
            st.subheader("📍 Sentiment by Platform")
            plats = df[st.session_state.plat_col].unique()
            for i in range(0, len(plats), 4):
                cols = st.columns(4)
                for j, p in enumerate(plats[i:i+4]):
                    with cols[j]:
                        pc = df[df[st.session_state.plat_col] == p]['Sentiment'].value_counts()
                        fig = px.pie(pc, values=pc.values, names=pc.index, title=f"<b>{p}</b>", height=180, color=pc.index, color_discrete_map=AESTHETIC_COLORS)
                        fig.update_layout(showlegend=False, margin=dict(l=5, r=5, t=30, b=0))
                        st.plotly_chart(fig, use_container_width=True, key=f"plat_{p}_{i}_{j}")

        st.markdown("---")
        st.subheader("☁️ Contextual WordClouds")
        wcols = st.columns(3)
        colors = ['Greens', 'Blues', 'Reds']
        for i, label in enumerate(['Positive', 'Neutral', 'Negative']):
            text_data = " ".join(df[df['Sentiment'] == label]['Cleaned']).strip()
            if text_data:
                wc = WordCloud(width=500, height=300, background_color='white', colormap=colors[i]).generate(text_data)
                wcols[i].image(wc.to_array(), caption=f"{label} Sentiment Keywords", use_container_width=True)
            else:
                wcols[i].info(f"Not enough data for {label}.")

elif selection == "📊 Product Sentiment":
    st.subheader("📊 Product Sentiment Analysis")
    du = st.session_state.data
    if du is not None:
        pcol = st.selectbox("Select Product Name Column", du.columns)
        unique_prods = du[pcol].dropna().unique()
        
        st.markdown("### 📈 Overall Product Sentiment")
        stats = []
        for p in unique_prods:
            pdf = du[du[pcol] == p]
            v = pdf['Sentiment'].value_counts()
            score = round(((v.get('Positive',0)-v.get('Negative',0))/len(pdf))*100, 2)
            stats.append({"Product": str(p), "Net Satisfaction (%)": score})
        
        st.plotly_chart(px.bar(pd.DataFrame(stats), x="Product", y="Net Satisfaction (%)", color="Net Satisfaction (%)", color_continuous_scale="RdYlGn", text_auto=True), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🥧 Individual Product Breakdown")
        icols = st.columns(3)
        for i, p in enumerate(unique_prods):
            with icols[i % 3]:
                pdf = du[du[pcol] == p]
                v = pdf['Sentiment'].value_counts()
                ind_score = round(((v.get('Positive',0)-v.get('Negative',0))/len(pdf))*100, 1)
                st.metric(label=f"Product: {str(p)}", value=f"{ind_score}% Score")
                fig_pie = px.pie(v, values=v.values, names=v.index, hole=0.4, height=250, color=v.index, color_discrete_map=AESTHETIC_COLORS)
                fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_chart_{str(p)}_{i}")

elif selection == "⚖️ Product Comparison":
    st.subheader("⚖️ Compare Two Products")
    c_file = st.file_uploader("Upload dataset for comparison", type="csv", key="comp_uploader")
    if c_file:
        dfc = pd.read_csv(c_file, encoding='latin1')
        dfc.columns = dfc.columns.str.strip()
        c1, c2 = st.columns(2)
        tx_col = c1.selectbox("Text Column", dfc.columns, key="c_tx")
        pd_col = c2.selectbox("Product Name Column", dfc.columns, key="c_pd")
        selected = st.multiselect("Select 2 Products", sorted(dfc[pd_col].dropna().astype(str).unique()))
        
        if st.button("Process Comparison") and len(selected) == 2:
            res = process_sentiment(dfc[dfc[pd_col].astype(str).isin(selected)].copy(), tx_col)
            st.session_state.c_data, st.session_state.c_prods, st.session_state.c_col_name = res, selected, pd_col
            
        if "c_data" in st.session_state:
            d, p_list, cn = st.session_state.c_data, st.session_state.c_prods, st.session_state.c_col_name
            comp_cols = st.columns(2)
            scores = {}
            for i, p in enumerate(p_list):
                sub = d[d[cn].astype(str) == p]
                v = sub['Sentiment'].value_counts()
                sc = round(((v.get('Positive',0)-v.get('Negative',0))/len(sub))*100, 2)
                scores[p] = sc
                with comp_cols[i]:
                    st.plotly_chart(px.pie(v, values=v.values, names=v.index, title=f"Product: {p} ({sc}%)", hole=0.3, color=v.index, color_discrete_map=AESTHETIC_COLORS), use_container_width=True, key=f"comp_pie_{i}")
                    st.markdown(f"#### 🔍 Key Drivers for **{p}**")
                    for sent, color in zip(['Positive', 'Negative', 'Neutral'], ['#10B981', '#F43F5E', '#3B82F6']):
                        sent_text = " ".join(sub[sub['Sentiment'] == sent]['Cleaned']).split()
                        common_words = [word for word, count in Counter(sent_text).most_common(5)]
                        if common_words:
                            st.markdown(f"<span style='color:{color}; font-weight:bold;'>{sent}:</span> {', '.join(common_words)}", unsafe_allow_html=True)

            winner = p_list[0] if scores[p_list[0]] > scores[p_list[1]] else p_list[1]
            st.success(f"**AI Result:** {winner} has a superior brand reputation.")

elif selection == "✍️ Manual Analysis":
    st.subheader("✍️ Real-time Prediction")
    text = st.text_area("Enter feedback:")
    if st.button("Analyze") and text:
        clean = clean_text(text)
        seq = pad_sequences(tokenizer.texts_to_sequences([clean]), maxlen=100)
        label = ["Negative", "Neutral", "Positive"][np.argmax(model.predict(seq))]
        st.markdown(f"### Predicted Sentiment: **{label}**")

elif selection == "🚨 Review Analysis":
    st.subheader("🚨 Priority Complaint Analysis")
    if st.session_state.data is not None:
        neg_df = st.session_state.data[st.session_state.data['Sentiment'] == 'Negative'].copy()
        if not neg_df.empty:
            prod_col_ref = st.selectbox("Identify Product Column", st.session_state.data.columns, key="rev_prod_ref")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Top Negative Products**")
                st.plotly_chart(px.bar(neg_df[prod_col_ref].value_counts().reset_index(), x=prod_col_ref, y='count', color_discrete_sequence=['#F43F5E']), use_container_width=True)
            with c2:
                st.write("**Negative Volume by Platform**")
                st.plotly_chart(px.bar(neg_df[st.session_state.plat_col].value_counts().reset_index(), x=st.session_state.plat_col, y='count', color_discrete_sequence=['#8B0000']), use_container_width=True)
            
            st.markdown("### 📝 Frequent Issue Log")
            top_complaints = neg_df[st.session_state.text_col].value_counts().reset_index().head(10)
            for i, row in top_complaints.iterrows():
                sample = neg_df[neg_df[st.session_state.text_col] == row[st.session_state.text_col]].iloc[0]
                st.markdown(f"<div class='priority-card'><b>Issue #{i+1} ({row['count']} Reports)</b><br><small>📍 {sample[st.session_state.plat_col]} | 📦 {sample[prod_col_ref]}</small><hr><i>\"{row[st.session_state.text_col]}\"</i></div>", unsafe_allow_html=True)
        else: st.success("✨ No negative reviews found!")
    else: st.warning("Upload data first.")

elif selection == "🔥 Brand Crisis Detection":
    st.subheader("🔥 Brand Crisis Detection")
    if st.session_state.data is not None:
        df = st.session_state.data
        neg_p = (len(df[df['Sentiment'] == 'Negative']) / len(df)) * 100
        if neg_p > 35:
            st.markdown(f"<div style='background:#721c24;color:white;padding:25px;border-radius:10px;text-align:center;'><h1>🚨 EMERGENCY</h1><p>Negative sentiment is <b>{neg_p:.1f}%</b></p></div>", unsafe_allow_html=True)
        else: st.success(f"✅ Stable. Negative sentiment at {neg_p:.1f}%.")
        
        keywords = ['scam', 'fake', 'dangerous', 'lawsuit', 'refund', 'worst', 'broken']
        critical = df[(df['Sentiment'] == 'Negative') & (df[st.session_state.text_col].str.lower().str.contains('|'.join(keywords)))]
        if not critical.empty:
            prod_ref = st.selectbox("Product Column", df.columns, key="cr_ref")
            display_cols = list(dict.fromkeys([prod_ref, st.session_state.plat_col, st.session_state.text_col]))
            st.dataframe(critical[display_cols], use_container_width=True)
        else: st.info("🔍 No high-risk keywords found.")

    else: st.warning("Upload data first.")
