# /full/path/to/your/project/market_research_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Market Research & Analysis Tool",
    page_icon="📊",
    layout="wide"
)

# --- Helper Functions ---
def display_dataframe_info(df):
    """Displays comprehensive information about the uploaded DataFrame."""
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

# --- Main Application ---
def main():
    st.title("📊 Market Research & Analysis Dashboard")
    st.markdown("""
    Welcome to the Market Research & Analysis Dashboard!
    Upload your market data (e.g., sales figures, customer surveys, competitor information)
    in CSV format to get started.
    """)

    # --- Sidebar for Data Upload ---
    st.sidebar.header("📁 Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv",
                                             help="Upload your dataset in CSV format for analysis.")

    if uploaded_file is not None:
        try:
            # Specify common encodings to try
            encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            df = None
            for encoding in encodings_to_try:
                try:
                    # Reset file pointer before trying a new encoding
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    st.sidebar.success(f"File Uploaded Successfully (encoding: {encoding})!")
                    break # Exit loop if successful
                except UnicodeDecodeError:
                    continue # Try next encoding
            
            if df is None:
                st.sidebar.error("Failed to decode the CSV file with common encodings. Please check the file format and encoding.")
                st.error("Could not read the CSV file. Please ensure it's a valid CSV and try a different encoding if necessary.")
                return


            # --- Data Exploration Section ---
            st.header("1. 🔍 Data Exploration & Overview")
            display_dataframe_info(df)

            # --- Basic Visualization Section ---
            st.header("2. 📈 Data Visualization")
            st.markdown("Select columns to visualize patterns and relationships in your data.")

            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            if not numerical_cols and not categorical_cols:
                st.warning("No plottable columns (numerical or categorical) found in the uploaded data.")
            else:
                viz_tab1, viz_tab2 = st.tabs(["Univariate Analysis", "Bivariate Analysis"])

                with viz_tab1:
                    st.subheader("Single Variable Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        if numerical_cols:
                            selected_num_col_hist = st.selectbox(
                                "Select Numerical Column for Histogram:",
                                numerical_cols,
                                key="hist_num_select",
                                help="Histograms show the distribution of a numerical variable."
                            )
                            if selected_num_col_hist:
                                fig, ax = plt.subplots()
                                sns.histplot(df[selected_num_col_hist], kde=True, ax=ax, color="skyblue")
                                ax.set_title(f"Distribution of {selected_num_col_hist}")
                                ax.set_xlabel(selected_num_col_hist)
                                ax.set_ylabel("Frequency")
                                st.pyplot(fig)
                        else:
                            st.info("No numerical columns available for histogram.")

                    with col2:
                        if categorical_cols:
                            selected_cat_col_bar = st.selectbox(
                                "Select Categorical Column for Bar Chart:",
                                categorical_cols,
                                key="bar_cat_select",
                                help="Bar charts show the frequency of each category in a categorical variable."
                            )
                            if selected_cat_col_bar:
                                fig, ax = plt.subplots()
                                count_data = df[selected_cat_col_bar].value_counts().nlargest(15) # Show top 15
                                sns.barplot(x=count_data.index, y=count_data.values, ax=ax, palette="viridis")
                                ax.set_title(f"Frequency of {selected_cat_col_bar} (Top 15)")
                                ax.set_xlabel(selected_cat_col_bar)
                                ax.set_ylabel("Count")
                                plt.xticks(rotation=45, ha='right')
                                plt.tight_layout()
                                st.pyplot(fig)
                        else:
                            st.info("No categorical columns available for bar chart.")

                with viz_tab2:
                    st.subheader("Relationship Between Two Variables")
                    if len(numerical_cols) >= 2:
                        st.markdown("#### Scatter Plot (Numerical vs Numerical)")
                        x_axis = st.selectbox("Select X-axis (Numerical):", numerical_cols, key="scatter_x_select")
                        y_axis_options = [col for col in numerical_cols if col != x_axis]
                        if not y_axis_options: # Only one numerical column
                             st.info("You need at least two different numerical columns for a scatter plot.")
                        else:
                            y_axis = st.selectbox("Select Y-axis (Numerical):", y_axis_options, key="scatter_y_select")

                            hue_options = [None] + categorical_cols + [col for col in numerical_cols if col not in [x_axis, y_axis]]
                            hue_col = st.selectbox("Select Hue for coloring (Optional):", hue_options, key="scatter_hue_select",
                                                help="Color points by a third variable (categorical or numerical).")

                            if x_axis and y_axis:
                                fig, ax = plt.subplots()
                                sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=hue_col if hue_col else None, ax=ax, palette="coolwarm")
                                title = f"Scatter Plot: {x_axis} vs {y_axis}"
                                if hue_col:
                                    title += f" (Colored by {hue_col})"
                                ax.set_title(title)
                                st.pyplot(fig)
                    else:
                        st.info("At least two numerical columns are needed for a scatter plot.")

                    if numerical_cols and categorical_cols:
                        st.markdown("#### Box Plot (Numerical by Categorical)")
                        cat_col_box = st.selectbox("Select Categorical Column (X-axis):", categorical_cols, key="box_cat_select")
                        num_col_box = st.selectbox("Select Numerical Column (Y-axis):", numerical_cols, key="box_num_select")

                        if cat_col_box and num_col_box:
                            # Limit number of categories for readability
                            top_n_categories = df[cat_col_box].nunique()
                            if top_n_categories > 10:
                                st.warning(f"'{cat_col_box}' has {top_n_categories} unique values. Displaying box plot for the most frequent 10.")
                                top_categories = df[cat_col_box].value_counts().nlargest(10).index
                                plot_df = df[df[cat_col_box].isin(top_categories)]
                            else:
                                plot_df = df

                            fig, ax = plt.subplots()
                            sns.boxplot(data=plot_df, x=cat_col_box, y=num_col_box, ax=ax, palette="pastel")
                            ax.set_title(f"Box Plot of {num_col_box} by {cat_col_box}")
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)


            # --- Placeholder Sections for Market Research ---
            st.header("3. 🛠️ Market Research Modules")
            st.markdown("These modules provide frameworks and tools for specific market analyses. (Currently placeholders, ready for development!)")

            with st.expander("📈 Trend Analysis"):
                st.markdown("""
                This section will allow you to:
                - Identify market trends from time-series data.
                - Visualize sales trends, customer acquisition rates, website traffic, etc.
                - **Future Enhancements:** Integrate with Google Trends API, apply smoothing techniques (e.g., moving averages), or conduct seasonality decomposition.
                """)
                date_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
                potential_date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]

                if not date_cols and potential_date_cols:
                    st.info(f"Found potential date/time columns: {', '.join(potential_date_cols)}. Consider converting them to datetime objects for time series analysis.")

                if date_cols or potential_date_cols:
                    chosen_date_col = st.selectbox("Select your primary Date/Time column:", date_cols + [col for col in potential_date_cols if col not in date_cols], key="trend_date_col")
                    if chosen_date_col:
                        try:
                            temp_df = df.copy()
                            temp_df[chosen_date_col] = pd.to_datetime(temp_df[chosen_date_col], errors='coerce')
                            temp_df = temp_df.dropna(subset=[chosen_date_col]) # Drop rows where date conversion failed

                            if not temp_df.empty and numerical_cols:
                                time_series_val = st.selectbox("Select value to plot over time:", numerical_cols, key="ts_val_select")
                                if time_series_val:
                                    agg_func = st.selectbox("Aggregation function (if multiple values per date):", ["mean", "sum", "median", "count"], key="ts_agg")
                                    
                                    # Resample if necessary (e.g., daily, weekly, monthly)
                                    resample_period = st.selectbox("Resample period (optional):", [None, "D", "W", "M", "Q", "Y"], key="ts_resample",
                                                                   format_func=lambda x: "None" if x is None else {"D":"Daily", "W":"Weekly", "M":"Monthly", "Q":"Quarterly", "Y":"Yearly"}[x])

                                    plot_data = temp_df.set_index(chosen_date_col)
                                    if resample_period:
                                        plot_data = plot_data[time_series_val].resample(resample_period).agg(agg_func)
                                    else:
                                        # If not resampling, group by date and aggregate if there are duplicates
                                        plot_data = plot_data.groupby(plot_data.index)[time_series_val].agg(agg_func)


                                    fig, ax = plt.subplots(figsize=(10, 5))
                                    plot_data.plot(ax=ax, marker='o', linestyle='-')
                                    ax.set_title(f"Time Series of {time_series_val} ({agg_func.capitalize()})")
                                    ax.set_ylabel(time_series_val)
                                    ax.set_xlabel(chosen_date_col)
                                    plt.grid(True)
                                    st.pyplot(fig)
                            else:
                                st.info("Ensure your selected date column is correctly formatted and you have numerical columns to plot.")
                        except Exception as e:
                            st.warning(f"Could not process '{chosen_date_col}' for time series analysis. Error: {e}")
                else:
                    st.info("Upload data with a clear 'Date' or 'Timestamp' column for trend analysis.")


            with st.expander("🎯 Competitor Analysis"):
                st.markdown("""
                This section helps you compare your offerings against competitors.
                - Analyze competitor pricing, features, market share, customer reviews, etc.
                - **Future Enhancements:** Tools for feature-by-feature comparison, automated data scraping (use ethically and respect terms of service), or sentiment analysis of competitor reviews.
                """)
                if len(df.columns) > 1:
                    st.write("Select columns relevant for competitor comparison (e.g., CompetitorName, Price, FeatureX, Rating):")
                    comp_cols = st.multiselect("Competitor attributes:", df.columns.tolist(), key="comp_attrs_select",
                                               help="Choose columns that represent competitor data.")
                    if comp_cols:
                        st.dataframe(df[comp_cols].head())
                        # You could add charts here, e.g., bar chart of average ratings per competitor if data allows
                else:
                    st.info("Upload data with multiple columns to perform competitor analysis.")


            with st.expander("👥 Customer Segmentation"):
                st.markdown("""
                This section will enable you to group customers based on shared characteristics.
                - **Methods:** Demographics (age, location), psychographics (lifestyle, values), behavior (purchase history, engagement).
                - **Future Enhancements:** Implement clustering algorithms (e.g., K-Means, DBSCAN from scikit-learn), visualize segments, and profile each segment.
                """)
                if numerical_cols:
                    st.info("For customer segmentation, you typically use numerical features like 'Age', 'Income', 'PurchaseFrequency', 'TimeOnSite'.")
                    st.write("Consider using clustering algorithms on selected features. For example, with K-Means:")
                    if len(numerical_cols) >= 2:
                        cluster_features = st.multiselect(
                            "Select numerical features for clustering (e.g., K-Means):",
                            numerical_cols,
                            key="cluster_feat_select",
                            help="Choose 2 or more numerical features for segmentation."
                        )
                        if len(cluster_features) >= 2:
                            num_clusters = st.slider("Number of clusters (K):", 2, 10, 3, key="kmeans_k")
                            st.write(f"You could apply K-Means clustering with K={num_clusters} to: {', '.join(cluster_features)}.")
                            st.markdown("*(Actual clustering implementation would require `scikit-learn` and further code.)*")
                    else:
                        st.info("You need at least two numerical features for most clustering algorithms.")
                else:
                    st.info("Upload data with numerical features to explore customer segmentation.")

            with st.expander("📊 SWOT Analysis (Strengths, Weaknesses, Opportunities, Threats)"):
                st.markdown("""
                A strategic planning framework to evaluate a company's competitive position and develop strategic initiatives.
                Use the text areas below to document your SWOT analysis based on your data and market insights.
                """)
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("Strengths (Internal, Positive):", height=150, key="swot_strengths",
                                 help="What does your company do well? What unique resources can you draw on?")
                    st.text_area("Weaknesses (Internal, Negative):", height=150, key="swot_weaknesses",
                                 help="What could you improve? Where do you have fewer resources than others?")
                with col2:
                    st.text_area("Opportunities (External, Positive):", height=150, key="swot_opportunities",
                                 help="What opportunities are open to you? What trends could you take advantage of?")
                    st.text_area("Threats (External, Negative):", height=150, key="swot_threats",
                                 help="What threats could harm you? What is your competition doing?")

            with st.expander("💬 Sentiment Analysis (e.g., from Customer Reviews)"):
                st.markdown("""
                Analyze text data (like customer reviews, social media comments, survey responses) to determine the underlying sentiment (positive, negative, neutral).
                - **Future Enhancements:** Integrate NLP libraries (e.g., NLTK, spaCy, Transformers by Hugging Face) for sentiment scoring, aspect-based sentiment analysis, and topic modeling.
                """)
                text_cols = df.select_dtypes(include=['object']).columns.tolist()
                if text_cols:
                    review_col = st.selectbox(
                        "Select column with text data (e.g., customer reviews, feedback):",
                        text_cols,
                        key="sentiment_col_select",
                        help="Choose a column containing text you want to analyze for sentiment."
                    )
                    if review_col:
                        st.info(f"You could apply sentiment analysis to the '{review_col}' column.")
                        st.write("Sample text data (first 5 entries):")
                        st.dataframe(df[[review_col]].head())
                        st.markdown("*(Actual sentiment analysis implementation would require NLP libraries and further code.)*")
                else:
                    st.info("Upload data with a text column (e.g., customer reviews) to explore sentiment analysis.")


        except pd.errors.EmptyDataError:
            st.sidebar.error("The uploaded file is empty. Please upload a valid CSV file.")
        except UnicodeDecodeError:
            st.sidebar.error("Encoding error. Please try uploading the CSV with UTF-8 encoding, or check if it's a valid CSV.")
            st.error("Could not decode the file. Please ensure it's a valid CSV and try saving it with UTF-8 encoding.")
        except Exception as e:
            st.sidebar.error(f"An error occurred during file processing: {e}")
            st.error(f"Could not process the uploaded file. Please ensure it's a valid CSV. Error details: {e}")
    else:
        st.info("👈 Upload a CSV file using the sidebar to begin your market research and analysis.")
        st.markdown("""
        ### What can this tool do?
        Once you upload your data, you'll be able to:
        1.  **Explore Your Data:** View summaries, data types, and identify missing values.
        2.  **Visualize Patterns:** Create interactive charts like histograms, bar charts, scatter plots, and box plots.
        3.  **Access Market Research Modules (Placeholders for deeper analysis):**
            *   **Trend Analysis:** Identify and visualize trends over time.
            *   **Competitor Analysis:** Compare against competitors on key metrics.
            *   **Customer Segmentation:** Group customers based on characteristics.
            *   **SWOT Analysis:** A framework for strategic planning.
            *   **Sentiment Analysis:** Understand opinions from text data.

        This is a foundational tool. You can expand these modules with more advanced algorithms and data sources!
        For example, you could integrate libraries like `scikit-learn` for machine learning, `nltk` or `spacy` for text processing, or `plotly` for more interactive visualizations.
        """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed with Streamlit.")


if __name__ == "__main__":
    main()

