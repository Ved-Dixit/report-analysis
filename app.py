# /Users/veddixit/market/app.py
import streamlit as st
import pandas as pd
import numpy as np # For numerical operations
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- New Imports for Added Features ---
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter # For summarization

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from duckduckgo_search import DDGS # For DuckDuckGo Search

# --- Page Configuration - MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(
    page_title="Market Research & Analysis Tool",
    page_icon="📊",
    layout="wide"
)

_nltk_download_messages = [] # Global list to store NLTK download messages

# --- Download NLTK resources (run once or handled by functions) ---
def download_nltk_resources():
    """Checks and downloads NLTK resources, collecting messages."""
    global _nltk_download_messages # Ensure we're modifying the global list
    resources = {
        "vader_lexicon": "sentiment/vader_lexicon.zip",
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords"
    }
    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
            _nltk_download_messages.append(f"NLTK resource '{resource_name}' already available.")
        except LookupError: # More general exception for NLTK download issues
            _nltk_download_messages.append(f"NLTK resource '{resource_name}' not found. Downloading...")
            try:
                nltk.download(resource_name, quiet=True)
                _nltk_download_messages.append(f"NLTK resource '{resource_name}' downloaded successfully.")
            except Exception as download_e: # Catch specific download errors
                _nltk_download_messages.append(f"Error downloading NLTK resource '{resource_name}': {download_e}")
        except Exception as e: # Catch other errors during find
            _nltk_download_messages.append(f"Error checking NLTK resource '{resource_name}': {e}")

download_nltk_resources() # Call this early to initiate downloads

# --- Helper Functions for Data Exploration (Existing) ---
def display_dataframe_info(df):
    st.subheader("DataFrame Overview")
    st.dataframe(df.head())
    st.subheader("DataFrame Shape")
    st.write(f"Number of Rows: {df.shape[0]}")
    st.write(f"Number of Columns: {df.shape[1]}")
    st.subheader("Data Types & Memory Usage")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.subheader("Descriptive Statistics")
    try:
        st.dataframe(df.describe(include='all'))
    except Exception as e:
        st.warning(f"Could not generate descriptive statistics for all columns. Error: {e}")
        st.dataframe(df.describe())
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    missing_df = missing_values[missing_values > 0].sort_values(ascending=False)
    if not missing_df.empty:
        st.dataframe(missing_df.to_frame(name='Missing Count'))
    else:
        st.success("🎉 No missing values found!")

# --- Helper Functions for New AI Features ---

# Web Scraping & Summarization
@st.cache_data(ttl=3600) # Cache for 1 hour
def scrape_website_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string if soup.title else "No title found"
        paragraphs = [p.get_text(separator=' ', strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
        full_text = "\n".join(paragraphs)
        return {"title": title, "paragraphs": paragraphs, "full_text": full_text}
    except requests.exceptions.RequestException as e:
        return {"error": f"Could not fetch URL: {e}"}
    except Exception as e:
        return {"error": f"An error occurred during scraping: {e}"}

@st.cache_data
def summarize_text_basic(text, num_sentences=3):
    if not text or not isinstance(text, str):
        return "Not enough text to summarize."
    try:
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return " ".join(sentences) # Return original text if it's short

        # Simple scoring: prioritize longer sentences with more common words (after stopword removal)
        stop_words = set(stopwords.words('english'))
        words = [word.lower() for word in word_tokenize(text) if word.isalpha() and word.lower() not in stop_words]
        if not words: # If only stopwords or no words
             return " ".join(sentences[:num_sentences])


        word_freq = Counter(words)
        most_common_words = set([word for word, freq in word_freq.most_common(10)]) # Top 10 common words as keywords

        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            sentence_words = [word.lower() for word in word_tokenize(sentence) if word.isalpha()]
            score = sum(1 for word in sentence_words if word in most_common_words) # Score by keyword presence
            score += len(sentence_words) * 0.01 # Slight preference for longer sentences
            sentence_scores[i] = score
        
        # Select top N sentences based on score, maintaining original order
        sorted_sentences_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
        top_indices = sorted(sorted_sentences_indices[:num_sentences])
        summary = " ".join([sentences[i] for i in top_indices])
        return summary
    except Exception as e:
        # Fallback to first N sentences if advanced summarization fails
        st.warning(f"Basic summarizer encountered an issue ({e}), falling back to simpler method.")
        sentences = text.split('.') # Simple split if sent_tokenize fails
        return ". ".join(sentences[:num_sentences]) + "."


# Sentiment Analysis
@st.cache_resource
def get_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text, analyzer):
    if not isinstance(text, str) or not text.strip():
        return {"label": "Neutral", "score": 0.0, "details": {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}}
    vs = analyzer.polarity_scores(text)
    if vs['compound'] >= 0.05:
        return {"label": "Positive", "score": vs['compound'], "details": vs}
    elif vs['compound'] <= -0.05:
        return {"label": "Negative", "score": vs['compound'], "details": vs}
    return {"label": "Neutral", "score": vs['compound'], "details": vs}

# Customer Segmentation (K-Means - from previous)
# No new helper function needed here, logic is in main app flow

# Topic Modeling
@st.cache_data(persist="disk") # Persist to disk for larger models/vectorizers
def perform_topic_modeling(texts, n_topics=5, n_top_words=10, max_df=0.95, min_df=2):
    if not texts or all(not t for t in texts):
        return None, None, "No valid text data provided for topic modeling."
    try:
        # Use CountVectorizer for LDA as it often works better than TF-IDF for LDA
        vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english', ngram_range=(1,1))
        tf = vectorizer.fit_transform(texts)
        
        if tf.shape[0] < n_topics: # Not enough documents for the number of topics
             return None, None, f"Number of documents ({tf.shape[0]}) is less than the number of topics ({n_topics}). Reduce topics or add more data."

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='online')
        lda.fit(tf)
        
        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        for topic_idx, topic_weights in enumerate(lda.components_):
            top_words_idx = topic_weights.argsort()[:-n_top_words - 1:-1]
            topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in top_words_idx]
        
        # Get topic distribution for each document
        doc_topic_dist = lda.transform(tf)
        return topics, doc_topic_dist, None
    except ValueError as ve: # Catch specific sklearn errors
        if "empty vocabulary" in str(ve):
            return None, None, "Topic modeling failed: Empty vocabulary. Check your text data and preprocessing (e.g. stopwords, min_df)."
        return None, None, f"Topic modeling failed with ValueError: {ve}. Try adjusting parameters like min_df or check data."
    except Exception as e:
        return None, None, f"An error occurred during topic modeling: {e}"


# Time Series Forecasting
@st.cache_data
def perform_time_series_forecasting(series, forecast_periods=12, model_type='additive', seasonal_periods=None):
    if not isinstance(series, pd.Series) or series.empty:
        return None, "Series is empty or not valid."
    if series.isnull().all():
        return None, "Series contains only NaN values."
    
    series = series.asfreq(series.index.inferred_freq or 'D').interpolate() # Ensure frequency and interpolate missing

    try:
        if model_type == 'simple':
            model = SimpleExpSmoothing(series, initialization_method="estimated").fit()
        elif model_type == 'additive' and seasonal_periods and seasonal_periods > 1 and len(series) > 2 * seasonal_periods :
            model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=seasonal_periods, initialization_method="estimated").fit()
        elif model_type == 'multiplicative' and seasonal_periods and seasonal_periods > 1 and len(series) > 2 * seasonal_periods:
             if (series <= 0).any(): # Multiplicative requires positive values
                return None, "Multiplicative seasonality requires all data points to be positive."
             model = ExponentialSmoothing(series, trend='add', seasonal='mul', seasonal_periods=seasonal_periods, initialization_method="estimated").fit()
        else: # Fallback to additive trend if seasonality is not appropriate or specified
            model = ExponentialSmoothing(series, trend='add', initialization_method="estimated").fit()
            if model_type not in ['simple']:
                 st.info(f"Using Exponential Smoothing with additive trend (no seasonality or seasonality criteria not met for '{model_type}').")


        forecast = model.forecast(forecast_periods)
        return forecast, None
    except Exception as e:
        return None, f"Error during forecasting: {e}. Try a simpler model or check data."

# DuckDuckGo Search
@st.cache_data(ttl=3600) # Cache search results for an hour
def search_duckduckgo(query, max_results=5):
    """Performs a DuckDuckGo search and returns results."""
    try:
        with DDGS() as ddgs:
            # ddgs.text returns a generator, convert to list
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return {"results": [], "message": "No results found for your query."}
        return {"results": results, "message": None}
    except Exception as e:
        return {"results": [], "message": f"Error during DuckDuckGo search: {e}"}

# --- Main Application ---
def main():
    st.title("📊 Advanced Market Research & AI Analysis Dashboard")
    st.markdown("""
    Welcome! Upload your market data (CSV) or use the tools below for analysis.
    This dashboard now includes web scraping, text summarization, enhanced sentiment analysis, 
    topic modeling, and time series forecasting.
    """)

    # Display NLTK download messages in the sidebar
    if _nltk_download_messages:
        with st.sidebar: # Ensure messages are placed in the sidebar
            st.subheader("NLTK Resource Status:")
            for msg in _nltk_download_messages:
                if "Error" in msg or "failed" in msg:
                    st.error(msg)
                elif "downloaded successfully" in msg or "Downloading" in msg :
                    st.success(msg)
                else:
                    st.info(msg)
            _nltk_download_messages.clear() # Clear messages after displaying

    # --- Sidebar for Data Upload ---
    st.sidebar.header("📁 Upload Your Data (CSV)")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv",
                                             help="Upload your dataset in CSV format for analysis.")
    df = None
    numerical_cols = [] # Initialize to avoid NameError if df is None
    categorical_cols = [] # Initialize to avoid NameError if df is None

    if uploaded_file is not None:
        try:
            encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for encoding in encodings_to_try:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    st.sidebar.success(f"File Uploaded Successfully (encoding: {encoding})!")
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                st.sidebar.error("Failed to decode CSV. Check file format/encoding.")
                st.error("Could not read CSV. Ensure valid format and try different encoding if necessary.")
                return # Stop execution if df is not loaded
        except Exception as e:
            st.sidebar.error(f"Error processing file: {e}")
            st.error(f"Could not process uploaded file. Error: {e}")
            return # Stop execution

    # --- Data Exploration & Visualization (Only if df is loaded) ---
    if df is not None:
        st.header("1. 🔍 Data Exploration & Overview")
        display_dataframe_info(df)

        st.header("2. 📈 Data Visualization")
        st.markdown("Select columns to visualize patterns and relationships in your data.")
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if not numerical_cols and not categorical_cols:
            st.warning("No plottable columns (numerical or categorical) found.")
        else:
            viz_tab1, viz_tab2 = st.tabs(["Univariate Analysis", "Bivariate Analysis"])
            with viz_tab1:
                st.subheader("Single Variable Analysis")
                if numerical_cols:
                    sel_num_hist = st.selectbox("Numerical Column for Histogram:", numerical_cols, key="snh")
                    if sel_num_hist:
                        fig, ax = plt.subplots()
                        sns.histplot(df[sel_num_hist], kde=True, ax=ax)
                        st.pyplot(fig)
                if categorical_cols:
                    sel_cat_bar = st.selectbox("Categorical Column for Bar Chart:", categorical_cols, key="scb")
                    if sel_cat_bar:
                        fig, ax = plt.subplots()
                        df[sel_cat_bar].value_counts().nlargest(10).plot(kind='bar', ax=ax)
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
            with viz_tab2:
                st.subheader("Relationship Between Two Variables")
                if len(numerical_cols) >= 2:
                    x_ax = st.selectbox("X-axis (Num):", numerical_cols, key="xax")
                    y_ax_opts = [c for c in numerical_cols if c != x_ax]
                    if y_ax_opts:
                        y_ax = st.selectbox("Y-axis (Num):", y_ax_opts, key="yax")
                        fig, ax = plt.subplots()
                        sns.scatterplot(data=df, x=x_ax, y=y_ax, ax=ax)
                        st.pyplot(fig)
                if numerical_cols and categorical_cols:
                    num_box = st.selectbox("Numerical for BoxPlot:", numerical_cols, key="nbx")
                    cat_box = st.selectbox("Categorical for BoxPlot:", categorical_cols, key="cbx")
                    fig, ax = plt.subplots()
                    sns.boxplot(data=df, x=cat_box, y=num_box, ax=ax)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)

        # --- Web Scraping & External Data (Now inside 'if df is not None') ---
        st.header("3. 🌐 Web Scraping & External Data")

        scrape_tab1, search_tab2 = st.tabs(["🔗 Scrape Specific URL", "🦆 Search with DuckDuckGo"])

        with scrape_tab1:
            st.markdown("Fetch basic information from a specific web page. *Always respect website terms of service.*")
            scrape_url_specific = st.text_input("Enter URL to scrape:", key="scrape_url_input_specific", help="e.g., https://www.example.com/article")
            if st.button("Scrape & Analyze URL", key="scrape_specific_button"):
                if scrape_url_specific:
                    with st.spinner(f"Scraping {scrape_url_specific}..."):
                        scraped_data = scrape_website_content(scrape_url_specific)
                    if "error" in scraped_data: # Check if the dictionary has an 'error' key
                        st.error(scraped_data["error"]) # Access the error message
                    else:
                        st.subheader(f"Scraped Title: {scraped_data['title']}")
                        st.markdown("#### Full Scraped Text (First 1000 chars):")
                        st.text_area("Full Text", scraped_data['full_text'][:1000]+"...", height=150, key="scraped_full_text_area")

                        if scraped_data['full_text']:
                            st.markdown("#### Basic Summary of Scraped Text:")
                            num_summary_sentences = st.slider("Number of sentences for summary:", 1, 10, 3, key="summary_sentences_slider")
                            with st.spinner("Generating summary..."):
                                summary = summarize_text_basic(scraped_data['full_text'], num_sentences=num_summary_sentences)
                            if summary:
                                st.success("Summary:")
                                st.write(summary)
                            else:
                                st.info("Could not generate summary.")
                        else:
                            st.info("No text content found in paragraphs to summarize.")
                else:
                    st.warning("Please enter a URL to scrape.")

        with search_tab2:
            st.markdown("Search the web using DuckDuckGo to find relevant articles, competitor information, or market trends.")
            search_query = st.text_input("Enter your search query for DuckDuckGo:", key="ddg_search_query")
            num_ddg_results = st.slider("Max number of search results:", 1, 20, 5, key="ddg_num_results")

            if st.button("Search DuckDuckGo", key="ddg_search_button"):
                if search_query:
                    with st.spinner(f"Searching DuckDuckGo for '{search_query}'..."):
                        search_data = search_duckduckgo(search_query, max_results=num_ddg_results)
                    
                    if search_data.get("message") and "Error" in search_data["message"]:
                        st.error(search_data["message"])
                    elif search_data.get("message"): # e.g., "No results found"
                        st.info(search_data["message"])
                    
                    if search_data.get("results"):
                        st.subheader(f"Search Results for '{search_query}':")
                        for i, result in enumerate(search_data["results"]):
                            st.markdown(f"#### {i+1}. {result.get('title', 'No Title')}")
                            st.markdown(f"**Link:** [{result.get('href', 'No URL')}]({result.get('href', '#')})")
                            st.markdown(f"**Snippet:** {result.get('body', 'No snippet available.')}")
                            
                            if result.get('href'):
                                scrape_key = f"scrape_ddg_result_{i}"
                                if st.button(f"Scrape & Summarize this result", key=scrape_key):
                                    with st.spinner(f"Scraping and summarizing {result.get('href')}..."):
                                        scraped_content_ddg = scrape_website_content(result.get('href'))
                                    if "error" in scraped_content_ddg:
                                        st.error(f"Could not scrape {result.get('href')}: {scraped_content_ddg['error']}")
                                    elif scraped_content_ddg.get('full_text'):
                                        summary_ddg = summarize_text_basic(scraped_content_ddg['full_text'], num_sentences=3)
                                        st.success(f"Summary for '{result.get('title', 'result')[:50]}...':")
                                        st.write(summary_ddg)
                                    else:
                                        st.info(f"No text content found at {result.get('href')} to summarize.")
                            st.markdown("---")
                else:
                    st.warning("Please enter a search query.")

        # --- AI-Powered Market Analysis Modules (Now inside 'if df is not None') ---
        st.header("4. 🧠 AI-Powered Market Analysis")

        with st.expander("📈 Trend Analysis & Forecasting"):
            st.markdown("Identify trends and forecast future values from time-series data.")
            # df is not None here
            date_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
            potential_date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'period' in col.lower()]
            all_potential_dates = list(set(date_cols + potential_date_cols))

            if not all_potential_dates:
                st.info("No clear date/time columns detected for trend analysis. Ensure your CSV has a date column.")
            else:
                chosen_date_col = st.selectbox("Select Date/Time column:", all_potential_dates, key="trend_date_col")
                if chosen_date_col and numerical_cols: # numerical_cols is defined if df is not None
                    try:
                        temp_df = df.copy()
                        temp_df[chosen_date_col] = pd.to_datetime(temp_df[chosen_date_col], errors='coerce')
                        temp_df = temp_df.dropna(subset=[chosen_date_col]).set_index(chosen_date_col)
                        
                        time_series_val = st.selectbox("Select value to plot/forecast:", numerical_cols, key="ts_val_select")
                        if time_series_val and time_series_val in temp_df.columns:
                            series_to_plot = temp_df[time_series_val].dropna()
                            
                            if series_to_plot.empty:
                                st.warning(f"No valid data in '{time_series_val}' after processing dates.")
                            else:
                                st.markdown("#### Time Series Plot")
                                fig_ts, ax_ts = plt.subplots(figsize=(10,5))
                                series_to_plot.plot(ax=ax_ts, marker='.', linestyle='-')
                                ax_ts.set_title(f"Time Series of {time_series_val}")
                                st.pyplot(fig_ts)

                                st.markdown("#### Forecasting")
                                forecast_periods = st.slider("Periods to forecast:", 1, 36, 12, key="fc_periods")
                                
                                freq = pd.infer_freq(series_to_plot.index)
                                seasonal_periods_map = {'D':7, 'W':52, 'M':12, 'Q':4, 'A':1, 'Y':1}
                                inferred_seasonal_periods = None
                                if freq:
                                    freq_prefix = freq.split('-')[0]
                                    inferred_seasonal_periods = seasonal_periods_map.get(freq_prefix.upper())
                                
                                seasonal_options = ['simple', 'additive']
                                if inferred_seasonal_periods and (series_to_plot > 0).all():
                                    seasonal_options.append('multiplicative')

                                forecast_model_type = st.selectbox("Forecast Model Type:", seasonal_options, key="fc_model_type",
                                                                   help="Simple: No trend/seasonality. Additive/Multiplicative: With trend and seasonality.")
                                
                                custom_seasonal_periods = inferred_seasonal_periods
                                if forecast_model_type != 'simple':
                                    custom_seasonal_periods = st.number_input("Seasonal Periods (e.g., 12 for monthly, 7 for daily):", 
                                                                              min_value=2, value=inferred_seasonal_periods or 12, key="custom_sp")

                                if st.button("Generate Forecast", key="gen_fc_btn"):
                                    with st.spinner("Forecasting..."):
                                        forecast_values, error_msg = perform_time_series_forecasting(
                                            series_to_plot, 
                                            forecast_periods, 
                                            model_type=forecast_model_type,
                                            seasonal_periods=custom_seasonal_periods if forecast_model_type != 'simple' else None
                                        )
                                    if error_msg:
                                        st.error(error_msg)
                                    elif forecast_values is not None:
                                        st.success("Forecast generated!")
                                        fig_fc, ax_fc = plt.subplots(figsize=(10,5))
                                        series_to_plot.plot(ax=ax_fc, label="Actual")
                                        forecast_values.plot(ax=ax_fc, label="Forecast", linestyle='--')
                                        ax_fc.set_title(f"Forecast of {time_series_val}")
                                        plt.legend()
                                        st.pyplot(fig_fc)
                                        st.write("Forecasted Values:")
                                        st.dataframe(forecast_values)
                                    else:
                                        st.error("Forecasting failed. Check data and parameters.")
                    except Exception as e:
                        st.error(f"Error in Trend Analysis: {e}")
            # else: # This else corresponds to `if df is not None and not df.empty:`
            #     st.info("Upload CSV data with date and numerical columns for Trend Analysis & Forecasting.") # Redundant as df is not None

        with st.expander("👥 Customer Segmentation (K-Means)"):
            st.markdown("Group customers based on shared characteristics using K-Means clustering.")
            if numerical_cols: # df is already confirmed not None, numerical_cols is defined
                if len(numerical_cols) >= 2:
                    cluster_features = st.multiselect(
                        "Select numerical features for clustering:",
                        numerical_cols,
                        default=numerical_cols[:2] if len(numerical_cols) >= 2 else None,
                        key="cluster_feat_select"
                    )
                    if len(cluster_features) >= 2:
                        num_clusters = st.slider("Number of clusters (K):", 2, 10, 3, key="kmeans_k")
                        if st.button("Perform K-Means Clustering", key="run_kmeans"):
                            with st.spinner("Running K-Means..."):
                                X = df[cluster_features].copy().dropna()
                                if X.empty:
                                    st.warning("No data after dropping NaNs from selected features.")
                                else:
                                    scaler = StandardScaler()
                                    X_scaled = scaler.fit_transform(X)
                                    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
                                    X['Cluster'] = kmeans.fit_predict(X_scaled)
                                    st.success(f"K-Means clustering complete! {num_clusters} clusters identified.")
                                    st.dataframe(X.head())
                                    fig_cluster, ax_cluster = plt.subplots()
                                    sns.scatterplot(data=X, x=cluster_features[0], y=cluster_features[1], hue='Cluster', palette='viridis', ax=ax_cluster)
                                    ax_cluster.set_title(f'Customer Segments ({cluster_features[0]} vs {cluster_features[1]})')
                                    st.pyplot(fig_cluster)
                                    st.markdown("#### Cluster Profiles (Mean Values):")
                                    st.dataframe(X.groupby('Cluster')[cluster_features].mean())
                    else:
                        st.info("Select at least two numerical features for clustering.")
                else:
                    st.info("Need at least two numerical columns for K-Means clustering.")
            else:
                st.info("No numerical columns found in the uploaded data for Customer Segmentation.")

        with st.expander("💬 Sentiment Analysis (VADER)"):
            st.markdown("Analyze text data for sentiment (positive, negative, neutral).")
            text_cols = df.select_dtypes(include=['object']).columns.tolist() # df is not None
            if text_cols:
                review_col = st.selectbox("Select text column for Sentiment Analysis:", text_cols, key="sentiment_col_select")
                if review_col:
                    analyzer = get_sentiment_analyzer()
                    if st.button("Analyze Full Column Sentiment", key="analyze_full_sentiment"):
                        with st.spinner(f"Analyzing sentiment for column '{review_col}'..."):
                            sentiments_data = df[review_col].dropna().astype(str).apply(lambda x: analyze_sentiment_vader(x, analyzer))
                            sentiment_labels = [s['label'] for s in sentiments_data]
                            sentiment_scores = [s['score'] for s in sentiments_data]
                        
                        if not sentiment_labels:
                            st.warning("No text data to analyze in the selected column after filtering.")
                        else:
                            st.success("Sentiment analysis complete!")
                            
                            sentiment_counts = pd.Series(sentiment_labels).value_counts()
                            fig_sent, ax_sent = plt.subplots()
                            sentiment_counts.plot(kind='pie', ax=ax_sent, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightgreen', 'lightskyblue'])
                            ax_sent.set_ylabel('')
                            ax_sent.set_title(f"Sentiment Distribution for '{review_col}'")
                            st.pyplot(fig_sent)

                            st.write("Overall Sentiment Scores (Compound):")
                            st.write(f"- Average Compound Score: {np.mean(sentiment_scores):.2f}")
                            st.write(f"- Median Compound Score: {np.median(sentiment_scores):.2f}")

                            results_df = pd.DataFrame({
                                "Text (Snippet)": df[review_col].dropna().astype(str).head(20).apply(lambda x: x[:100]+"..."),
                                "Sentiment": [s['label'] for s in sentiments_data.head(20)],
                                "Compound Score": [s['score'] for s in sentiments_data.head(20)]
                            })
                            st.dataframe(results_df)
            else:
                st.info("No text (object type) columns found in the uploaded data.")

        with st.expander("📚 Topic Modeling (LDA)"):
            st.markdown("Discover underlying topics in your text data using Latent Dirichlet Allocation.")
            text_cols_lda = df.select_dtypes(include=['object']).columns.tolist() # df is not None
            if text_cols_lda:
                lda_text_col = st.selectbox("Select text column for Topic Modeling:", text_cols_lda, key="lda_text_col_select")
                if lda_text_col:
                    n_topics = st.slider("Number of Topics:", 2, 15, 5, key="lda_n_topics")
                    n_top_words = st.slider("Number of Top Words per Topic:", 3, 15, 7, key="lda_n_top_words")
                    
                    st.markdown("###### Advanced Parameters (Optional)")
                    max_df_lda = st.slider("Max Document Frequency (max_df)", 0.50, 1.00, 0.95, 0.01, help="Ignore terms that appear in more than this fraction of documents.")
                    min_df_lda = st.slider("Min Document Frequency (min_df)", 1, 10, 2, help="Ignore terms that appear in less than this absolute number of documents.")

                    if st.button("Perform Topic Modeling", key="run_lda"):
                        texts_for_lda = df[lda_text_col].dropna().astype(str).tolist()
                        if not texts_for_lda:
                            st.warning("No text data available in the selected column after filtering.")
                        else:
                            with st.spinner("Running LDA Topic Modeling... This may take a moment."):
                                topics, doc_topic_dist, error_msg = perform_topic_modeling(
                                    texts_for_lda, n_topics, n_top_words, max_df=max_df_lda, min_df=min_df_lda
                                )
                            if error_msg:
                                st.error(error_msg)
                            elif topics:
                                st.success("Topic Modeling Complete!")
                                for topic_name, words in topics.items():
                                    st.subheader(topic_name)
                                    st.write(", ".join(words))
                                
                                if doc_topic_dist is not None:
                                    st.markdown("---")
                                    st.markdown("###### Topic Distribution for Sample Documents (Top 5)")
                                    sample_doc_topics = pd.DataFrame(doc_topic_dist[:5], columns=[f"Topic {i+1}" for i in range(n_topics)])
                                    sample_doc_topics.index = [f"Doc {i+1}" for i in range(min(5, len(texts_for_lda)))]
                                    st.dataframe(sample_doc_topics.style.format("{:.2%}"))
                            else:
                                st.error("Topic modeling failed to produce results.")
            else:
                st.info("No text (object type) columns found for Topic Modeling.")

        with st.expander("📊 SWOT Analysis"):
            st.markdown("A strategic planning framework. Input your analysis based on data and insights.")
            col1, col2 = st.columns(2)
            with col1: st.text_area("Strengths:", height=100, key="swot_s")
            with col1: st.text_area("Weaknesses:", height=100, key="swot_w")
            with col2: st.text_area("Opportunities:", height=100, key="swot_o")
            with col2: st.text_area("Threats:", height=100, key="swot_t")

        with st.expander("🎯 Competitor Analysis"):
            st.markdown("Framework for comparing against competitors. (Further AI integration can be added).")
            if len(df.columns) > 1: # df is not None
                st.write("Select columns relevant for competitor comparison:")
                comp_cols = st.multiselect("Competitor attributes:", df.columns.tolist(), key="comp_attrs_select_v2")
                if comp_cols:
                    st.dataframe(df[comp_cols].head())
            else:
                st.info("Need data with multiple columns to perform competitor analysis.")
    else:
        # This block executes if df is None (either no upload or failed upload)
        if uploaded_file is not None: # Means an upload was attempted but df is still None (error)
            st.warning("Data could not be loaded. Please check the file format/encoding and try again.")
        
        # --- Initial Message / Footer when no data is loaded ---
        st.markdown("---")
        st.info("👈 Upload a CSV file using the sidebar to unlock all analysis tools.")
        st.markdown("""
        ### Tool Capabilities (available after CSV upload):
        *   **Data Exploration & Visualization:** Understand your CSV data.
        *   **Web Scraping & Summarization:** Fetch and summarize web content.
        *   **AI-Powered Analysis:**
            *   Trend Analysis & Forecasting
            *   Customer Segmentation (K-Means)
            *   Sentiment Analysis (VADER)
            *   Topic Modeling (LDA)
        *   **Strategic Frameworks:** SWOT Analysis, Competitor Analysis.
        """)
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed with Streamlit & AI.")

if __name__ == "__main__":
    main()
