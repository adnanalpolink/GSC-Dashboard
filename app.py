import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from google.oauth2 import service_account
from google.cloud import bigquery
import re
import os
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder
from collections import Counter
import nltk
from nltk.util import ngrams
import io
import base64
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# Page configuration
st.set_page_config(
    page_title="GSC BigQuery Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Google Search Console BigQuery Analyzer")

# Authentication functions
@st.cache_resource
def get_bigquery_client(credentials_file=None):
    """Create and cache a BigQuery client."""
    try:
        if not credentials_file and 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
            credentials_file = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
        
        if credentials_file:
            credentials = service_account.Credentials.from_service_account_file(credentials_file)
            return bigquery.Client(credentials=credentials)
        
        # If credentials are uploaded via UI
        if 'bq_credentials' in st.session_state:
            credentials_json = st.session_state['bq_credentials']
            credentials = service_account.Credentials.from_service_account_info(credentials_json)
            return bigquery.Client(credentials=credentials)
            
        return None
    except Exception as e:
        st.error(f"Error creating BigQuery client: {e}")
        return None

def setup_authentication():
    """Setup BigQuery authentication."""
    with st.sidebar.expander("📝 BigQuery Authentication", expanded='bq_client' not in st.session_state):
        auth_method = st.radio(
            "Authentication Method",
            ["Upload Service Account JSON", "Use Environment Variable"]
        )
        
        if auth_method == "Upload Service Account JSON":
            credentials_file = st.file_uploader("Upload GCP Service Account JSON", type=['json'])
            if credentials_file is not None:
                credentials_json = json.loads(credentials_file.getvalue().decode('utf-8'))
                st.session_state['bq_credentials'] = credentials_json
                st.session_state['bq_client'] = get_bigquery_client()
                st.success("✅ BigQuery client created successfully!")
        else:
            creds_path = st.text_input("Path to credentials file (or set as GOOGLE_APPLICATION_CREDENTIALS env var)")
            if st.button("Connect"):
                if creds_path:
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
                st.session_state['bq_client'] = get_bigquery_client()
                if st.session_state['bq_client']:
                    st.success("✅ BigQuery client created successfully!")

# Data pipeline functions
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gsc_data(project_id, dataset_id, table_id, start_date, end_date, limit=None):
    """Fetch Google Search Console data from BigQuery."""
    client = st.session_state.get('bq_client')
    if not client:
        st.error("BigQuery client not initialized. Please set up authentication.")
        return None
    
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    query = f"""
    SELECT 
        date, query, page, country, device, search_type,
        impressions, clicks, ctr, position
    FROM `{project_id}.{dataset_id}.{table_id}`
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    {limit_clause}
    """
    
    try:
        with st.spinner("Fetching data from BigQuery..."):
            df = client.query(query).to_dataframe()
            return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def save_to_parquet(df, filename="gsc_data.parquet"):
    """Save DataFrame to Parquet file."""
    if df is not None:
        path = Path(filename)
        try:
            table = pa.Table.from_pandas(df)
            pq.write_table(table, path)
            return path
        except Exception as e:
            st.error(f"Error saving to Parquet: {e}")
    return None

def load_from_parquet(filename="gsc_data.parquet"):
    """Load DataFrame from Parquet file."""
    path = Path(filename)
    if path.exists():
        try:
            table = pq.read_table(path)
            return table.to_pandas()
        except Exception as e:
            st.error(f"Error loading from Parquet: {e}")
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def process_data_chunks(df, chunk_size=10000):
    """Process large DataFrame in chunks."""
    if df is None or len(df) == 0:
        return df
    
    processed_chunks = []
    
    with st.spinner("Processing data in chunks..."):
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size].copy()
            # Process chunk (add any transformations here)
            processed_chunks.append(chunk)
    
    return pd.concat(processed_chunks, ignore_index=True)

# Analysis functions
@st.cache_data(ttl=3600, show_spinner=False)
def generate_ngrams(series, n=2, weight_by=None, min_freq=3):
    """Generate n-grams from a series of queries."""
    if series.empty:
        return pd.Series()
    
    if weight_by is not None and weight_by in series.index.names:
        weighted = True
    else:
        weighted = False
    
    all_ngrams = Counter()
    
    for text in series.index:
        if isinstance(text, str):
            tokens = text.lower().split()
            if len(tokens) >= n:
                text_ngrams = list(ngrams(tokens, n))
                if weighted:
                    weight = series.loc[text]
                    for ng in text_ngrams:
                        all_ngrams[' '.join(ng)] += weight
                else:
                    for ng in text_ngrams:
                        all_ngrams[' '.join(ng)] += 1
    
    # Filter by minimum frequency
    all_ngrams = {k: v for k, v in all_ngrams.items() if v >= min_freq}
    
    return pd.Series(all_ngrams).sort_values(ascending=False)

def calculate_trend(df, metric, date_col='date'):
    """Calculate trend slope for a given metric."""
    if df.empty or metric not in df.columns or date_col not in df.columns:
        return 0, []
    
    # Aggregate by date
    trend_data = df.groupby(date_col)[metric].sum().reset_index()
    
    # Convert to numeric for calculation
    x = np.arange(len(trend_data))
    y = trend_data[metric].values
    
    # Calculate slope
    if len(y) >= 2:
        slope = np.polyfit(x, y, 1)[0]
        # Normalize by the mean value
        normalized_slope = (slope / np.mean(y)) * 100
        return normalized_slope, trend_data
    
    return 0, trend_data

def detect_cannibalization(df, query_threshold=2, click_threshold=10):
    """Detect potential query cannibalization."""
    if df.empty or 'query' not in df.columns or 'page' not in df.columns or 'clicks' not in df.columns:
        return pd.DataFrame()
    
    # Group by query and page, summing clicks
    cannibalization_df = df.groupby(['query', 'page']).agg({'clicks': 'sum'}).reset_index()
    
    # Count occurrences of each query
    query_counts = cannibalization_df['query'].value_counts()
    
    # Filter queries that appear in multiple pages
    multi_page_queries = query_counts[query_counts >= query_threshold].index
    cannibalization_candidates = cannibalization_df[cannibalization_df['query'].isin(multi_page_queries)]
    
    # Filter for meaningful traffic (click threshold)
    cannibalization_candidates = cannibalization_candidates[cannibalization_candidates['clicks'] >= click_threshold]
    
    # Sort by query and clicks (descending)
    if not cannibalization_candidates.empty:
        cannibalization_candidates = cannibalization_candidates.sort_values(['query', 'clicks'], ascending=[True, False])
    
    return cannibalization_candidates

def create_query_classification(df, patterns):
    """Classify queries based on regex patterns."""
    if df.empty or 'query' not in df.columns:
        return df
    
    result = df.copy()
    result['category'] = 'Uncategorized'
    
    for category, pattern in patterns.items():
        if pattern:
            try:
                mask = result['query'].str.contains(pattern, case=False, regex=True)
                result.loc[mask, 'category'] = category
            except re.error:
                st.warning(f"Invalid regex pattern for {category}: {pattern}")
    
    return result

# UI Components
def create_sidebar():
    """Create sidebar controls."""
    with st.sidebar:
        st.header("Data Controls")
        
        # Project and dataset settings
        with st.expander("BigQuery Settings", expanded=False):
            project_id = st.text_input("BigQuery Project ID", "your-project-id")
            dataset_id = st.text_input("Dataset ID", "searchconsole")
            table_id = st.text_input("Table ID", "gsc_data")
        
        # Date range selector
        today = datetime.now().date()
        default_start = today - timedelta(days=30)
        date_range = st.date_input(
            "Date Range",
            [default_start, today],
            min_value=today - timedelta(days=365),
            max_value=today
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = default_start
            end_date = today
        
        # Metric selector
        metric = st.selectbox(
            "Primary Metric",
            ["impressions", "clicks", "ctr", "position"],
            format_func=lambda x: x.title()
        )
        
        # Dimension selector
        dimensions = st.multiselect(
            "Split By",
            ["country", "device", "search_type", "page"],
            default=["country"],
            format_func=lambda x: x.title()
        )
        
        # Filters
        regex_filter = st.text_input("Query/URL Regex Filter", "")
        min_impressions = st.number_input("Min Impressions", 0, 10000, 100)
        min_clicks = st.number_input("Min Clicks", 0, 1000, 0)
        
        # Fetch data button
        fetch_btn = st.button("Fetch Data", use_container_width=True)
        
        # Sample data option
        use_sample = st.checkbox("Use sample data (faster)")
        sample_size = st.slider("Sample Size", 1000, 100000, 10000, 1000) if use_sample else None
        
        return {
            'project_id': project_id,
            'dataset_id': dataset_id,
            'table_id': table_id,
            'start_date': start_date,
            'end_date': end_date,
            'metric': metric,
            'dimensions': dimensions,
            'regex_filter': regex_filter,
            'min_impressions': min_impressions,
            'min_clicks': min_clicks,
            'fetch_data': fetch_btn,
            'use_sample': use_sample,
            'sample_size': sample_size
        }

def apply_filters(df, filters):
    """Apply filters to the DataFrame."""
    if df is None or df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Apply regex filter
    if filters['regex_filter']:
        try:
            query_mask = filtered_df['query'].str.contains(filters['regex_filter'], case=False, regex=True)
            page_mask = filtered_df['page'].str.contains(filters['regex_filter'], case=False, regex=True)
            filtered_df = filtered_df[query_mask | page_mask]
        except re.error:
            st.warning(f"Invalid regex pattern: {filters['regex_filter']}")
    
    # Apply min impressions and clicks
    filtered_df = filtered_df[
        (filtered_df['impressions'] >= filters['min_impressions']) &
        (filtered_df['clicks'] >= filters['min_clicks'])
    ]
    
    return filtered_df

def create_main_dashboard(df, filters):
    """Create the main dashboard UI."""
    if df is None or df.empty:
        st.warning("No data to display. Please fetch data first.")
        return
    
    filtered_df = apply_filters(df, filters)
    
    if filtered_df.empty:
        st.warning("No data matches the current filters.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", "Query Analysis", "Page Analysis", "Advanced Reports"
    ])
    
    # Tab 1: Overview
    with tab1:
        create_overview_tab(filtered_df, filters)
    
    # Tab 2: Query Analysis
    with tab2:
        create_query_analysis_tab(filtered_df, filters)
    
    # Tab 3: Page Analysis
    with tab3:
        create_page_analysis_tab(filtered_df, filters)
    
    # Tab 4: Advanced Reports
    with tab4:
        create_advanced_reports_tab(filtered_df, filters)

def create_overview_tab(df, filters):
    """Create the overview dashboard tab."""
    st.header("Performance Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_impressions = df['impressions'].sum()
        st.metric("Total Impressions", f"{total_impressions:,}")
    
    with col2:
        total_clicks = df['clicks'].sum()
        st.metric("Total Clicks", f"{total_clicks:,}")
    
    with col3:
        avg_ctr = (total_clicks / total_impressions) * 100 if total_impressions > 0 else 0
        st.metric("Average CTR", f"{avg_ctr:.2f}%")
    
    with col4:
        avg_position = df['position'].mean()
        st.metric("Average Position", f"{avg_position:.2f}")
    
    # Trend analysis
    st.subheader("Trend Analysis")
    
    trend_slope, trend_data = calculate_trend(df, filters['metric'])
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if not trend_data.empty:
            fig = px.line(
                trend_data, 
                x='date', 
                y=filters['metric'], 
                title=f"{filters['metric'].title()} Trend Over Time"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric(
            "Trend Direction", 
            f"{trend_slope:.2f}% daily change",
            delta=trend_slope,
            delta_color="inverse" if filters['metric'] == 'position' else "normal"
        )
        
        # Add some context
        if trend_slope > 0:
            trend_text = "increasing" if filters['metric'] != 'position' else "decreasing"
        elif trend_slope < 0:
            trend_text = "decreasing" if filters['metric'] != 'position' else "increasing"
        else:
            trend_text = "stable"
            
        st.caption(f"The {filters['metric']} is {trend_text} at a rate of {abs(trend_slope):.2f}% per day.")
    
    # Dimension breakdown
    st.subheader("Dimension Breakdown")
    
    selected_dimensions = st.multiselect(
        "Select dimensions to analyze",
        ["country", "device", "search_type", "page"],
        default=filters['dimensions'][:1] if filters['dimensions'] else ["country"]
    )
    
    if selected_dimensions:
        # Create a subplot for each selected dimension
        dimension_dfs = []
        
        for dim in selected_dimensions:
            dim_df = df.groupby(dim).agg({
                'impressions': 'sum',
                'clicks': 'sum',
                'ctr': 'mean',
                'position': 'mean'
            }).reset_index()
            
            dim_df = dim_df.sort_values(filters['metric'], ascending=(filters['metric'] == 'position'))
            dim_df = dim_df.head(10)  # Top 10 for readability
            
            dimension_dfs.append((dim, dim_df))
        
        # Display in grid of charts
        cols = st.columns(min(len(dimension_dfs), 2))
        
        for i, (dim, dim_df) in enumerate(dimension_dfs):
            col_idx = i % 2
            with cols[col_idx]:
                fig = px.bar(
                    dim_df,
                    x=dim,
                    y=filters['metric'],
                    title=f"Top {dim.title()} by {filters['metric'].title()}",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

def create_query_analysis_tab(df, filters):
    """Create the query analysis tab."""
    st.header("Query Analysis")
    
    # Top queries
    st.subheader("Top Queries")
    
    # Columns for controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        sort_by = st.selectbox(
            "Sort By",
            ["impressions", "clicks", "ctr", "position"],
            index=["impressions", "clicks", "ctr", "position"].index(filters['metric']),
            format_func=lambda x: x.title()
        )
    
    with col2:
        top_n = st.slider("Number of Queries", 5, 100, 20)
    
    with col3:
        additional_filter = st.text_input("Additional Query Filter (regex)", "")
    
    # Filter and sort data
    query_df = df.copy()
    
    if additional_filter:
        try:
            query_df = query_df[query_df['query'].str.contains(additional_filter, case=False, regex=True)]
        except re.error:
            st.warning(f"Invalid regex pattern: {additional_filter}")
    
    # Aggregate by query
    query_agg = query_df.groupby('query').agg({
        'impressions': 'sum',
        'clicks': 'sum',
        'ctr': 'mean',
        'position': 'mean'
    }).reset_index()
    
    # Sort and take top N
    is_ascending = sort_by == 'position'  # Lower position is better
    query_agg = query_agg.sort_values(sort_by, ascending=is_ascending).head(top_n)
    
    # Display as interactive table
    gb = GridOptionsBuilder.from_dataframe(query_agg)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
    gb.configure_column('query', headerName='Query', minWidth=250)
    gb.configure_column('impressions', headerName='Impressions', type='numericColumn', valueFormatter='d3.format(",d")(data.impressions)')
    gb.configure_column('clicks', headerName='Clicks', type='numericColumn', valueFormatter='d3.format(",d")(data.clicks)')
    gb.configure_column('ctr', headerName='CTR', type='numericColumn', valueFormatter='(data.ctr * 100).toFixed(2) + "%"')
    gb.configure_column('position', headerName='Position', type='numericColumn', valueFormatter='data.position.toFixed(1)')
    
    grid_options = gb.build()
    
    st.text("Click on column headers to sort")
    AgGrid(query_agg, gridOptions=grid_options, height=400)
    
    # N-gram Analysis
    st.subheader("N-gram Analysis")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        n = st.slider("N-gram Size", 1, 5, 2)
    
    with col2:
        weight_col = st.selectbox(
            "Weight By",
            [None, "impressions", "clicks"],
            format_func=lambda x: "Count" if x is None else x.title()
        )
    
    with col3:
        min_freq = st.slider("Minimum Frequency", 1, 100, 3)
    
    # Prepare series for n-gram analysis
    if weight_col:
        ngram_series = pd.Series(
            query_df[weight_col].values,
            index=query_df['query'].values
        )
    else:
        ngram_series = pd.Series(1, index=query_df['query'].values)
    
    # Generate n-grams
    with st.spinner(f"Generating {n}-grams..."):
        ngrams_df = generate_ngrams(ngram_series, n=n, min_freq=min_freq)
    
    if not ngrams_df.empty:
        # Display as bar chart
        fig = px.bar(
            ngrams_df.reset_index().head(20),
            x='index',
            y=0,
            labels={'index': f'{n}-gram', 0: 'Frequency' if weight_col is None else weight_col.title()},
            title=f"Top {n}-grams by {'Count' if weight_col is None else weight_col.title()}",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No n-grams found with the current settings.")

def create_page_analysis_tab(df, filters):
    """Create the page analysis tab."""
    st.header("Page Analysis")
    
    # Top pages
    st.subheader("Top Pages")
    
    # Columns for controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        sort_by = st.selectbox(
            "Sort Pages By",
            ["impressions", "clicks", "ctr", "position"],
            index=["impressions", "clicks", "ctr", "position"].index(filters['metric']),
            format_func=lambda x: x.title(),
            key="page_sort"
        )
    
    with col2:
        top_n = st.slider("Number of Pages", 5, 100, 20, key="page_top_n")
    
    with col3:
        page_filter = st.text_input("Page Filter (regex)", "")
    
    # Filter and sort data
    page_df = df.copy()
    
    if page_filter:
        try:
            page_df = page_df[page_df['page'].str.contains(page_filter, case=False, regex=True)]
        except re.error:
            st.warning(f"Invalid regex pattern: {page_filter}")
    
    # Aggregate by page
    page_agg = page_df.groupby('page').agg({
        'impressions': 'sum',
        'clicks': 'sum',
        'ctr': 'mean',
        'position': 'mean'
    }).reset_index()
    
    # Sort and take top N
    is_ascending = sort_by == 'position'  # Lower position is better
    page_agg = page_agg.sort_values(sort_by, ascending=is_ascending).head(top_n)
    
    # Display as interactive table
    gb = GridOptionsBuilder.from_dataframe(page_agg)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
    gb.configure_column('page', headerName='Page URL', minWidth=300)
    gb.configure_column('impressions', headerName='Impressions', type='numericColumn', valueFormatter='d3.format(",d")(data.impressions)')
    gb.configure_column('clicks', headerName='Clicks', type='numericColumn', valueFormatter='d3.format(",d")(data.clicks)')
    gb.configure_column('ctr', headerName='CTR', type='numericColumn', valueFormatter='(data.ctr * 100).toFixed(2) + "%"')
    gb.configure_column('position', headerName='Position', type='numericColumn', valueFormatter='data.position.toFixed(1)')
    
    grid_options = gb.build()
    
    st.text("Click on column headers to sort")
    AgGrid(page_agg, gridOptions=grid_options, height=400)
    
    # Page/Query Distribution
    st.subheader("Queries per Page Distribution")
    
    # Count queries per page
    query_counts = page_df.groupby('page')['query'].nunique().reset_index()
    query_counts.columns = ['page', 'query_count']
    query_counts = query_counts.sort_values('query_count', ascending=False).head(20)
    
    if not query_counts.empty:
        fig = px.bar(
            query_counts,
            x='page',
            y='query_count',
            title="Pages with Most Unique Queries",
            labels={'page': 'Page URL', 'query_count': 'Unique Query Count'},
            height=400
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Path distribution
    st.subheader("URL Path Analysis")
    
    # Extract path components
    page_df['path'] = page_df['page'].str.extract(r'https?://[^/]+(/[^?#]*)')
    page_df['path_depth'] = page_df['path'].str.count('/')
    
    # Group by path depth
    path_depth_df = page_df.groupby('path_depth').agg({
        'impressions': 'sum',
        'clicks': 'sum',
        'page': 'nunique'
    }).reset_index()
    
    path_depth_df.columns = ['Path Depth', 'Impressions', 'Clicks', 'Unique Pages']
    
    if not path_depth_df.empty:
        # Create two charts side by side
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                path_depth_df,
                x='Path Depth',
                y=['Impressions', 'Clicks'],
                title="Performance by URL Depth",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                path_depth_df,
                x='Path Depth',
                y='Unique Pages',
                title="URL Count by Path Depth",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

def create_advanced_reports_tab(df, filters):
    """Create the advanced reports tab."""
    st.header("Advanced Reports")
    
    # Subtabs for different advanced reports
    subtab1, subtab2, subtab3 = st.tabs([
        "Query Cannibalization", "Query Classification", "Multi-Dimensional Analysis"
    ])
    
    # Subtab 1: Query Cannibalization
    with subtab1:
        st.subheader("Query Cannibalization Report")
        st.caption("Identifies queries that appear in multiple pages, potentially causing internal competition.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            query_threshold = st.slider("Min Pages per Query", 2, 10, 2)
        
        with col2:
            click_threshold = st.slider("Min Clicks per Query-Page", 1, 100, 10)
        
        # Detect cannibalization
        with st.spinner("Analyzing query cannibalization..."):
            cannibalization_df = detect_cannibalization(
                df, 
                query_threshold=query_threshold, 
                click_threshold=click_threshold
            )
        
        if not cannibalization_df.empty:
            # Display as interactive table
            gb = GridOptionsBuilder.from_dataframe(cannibalization_df)
            gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=15)
            gb.configure_column('query', headerName='Query', minWidth=200)
            gb.configure_column('page', headerName='Page URL', minWidth=300)
            gb.configure_column('clicks', headerName='Clicks', type='numericColumn')
            
            grid_options = gb.build()
            
            st.text("These queries appear on multiple pages - possible cannibalization")
            AgGrid(cannibalization_df, gridOptions=grid_options, height=500)
            
            # Summary of most cannibalized queries
            query_page_counts = cannibalization_df.groupby('query').size().reset_index()
            query_page_counts.columns = ['query', 'page_count']
            query_page_counts = query_page_counts.sort_values('page_count', ascending=False).head(10)
            
            st.subheader("Most Cannibalized Queries")
            fig = px.bar(
                query_page_counts,
                x='query',
                y='page_count',
                title="Queries with Most Competing Pages",
                labels={'query': 'Query', 'page_count': 'Page Count'},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No query cannibalization detected with current settings.")
    
    # Subtab 2: Query Classification
    with subtab2:
        st.subheader("Query Classification System")
        st.caption("Categorize queries using regex patterns to segment your search traffic.")
        
        # Default patterns
        if 'classification_patterns' not in st.session_state:
            st.session_state.classification_patterns = {
                "Branded": "",
                "Informational": "(how|what|why|when|who|guide|tutorial)",
                "Transactional": "(buy|price|cost|purchase|order|shop)",
                "Navigational": "(login|sign|account|contact)"
            }
        
        # Pattern editor
        with st.expander("Edit Classification Patterns", expanded=True):
            # Allow user to add/edit categories
            pattern_updates = {}
            
            for category, pattern in st.session_state.classification_patterns.items():
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.text(category)
                
                with col2:
                    pattern_updates[category] = st.text_input(
                        f"Regex for {category}",
                        value=pattern,
                        key=f"pattern_{category}",
                        label_visibility="collapsed"
                    )
            
            # Option to add new category
            new_category = st.text_input("Add New Category")
            new_pattern = st.text_input("Pattern for New Category")
            
            if st.button("Add Category") and new_category and new_pattern:
                pattern_updates[new_category] = new_pattern
            
            # Update patterns
            st.session_state.classification_patterns = pattern_updates
        
        # Apply classification
        if st.button("Classify Queries"):
            with st.spinner("Classifying queries..."):
                classified_df = create_query_classification(df, st.session_state.classification_patterns)
                
                if not classified_df.empty:
                    # Summarize by category
                    category_summary = classified_df.groupby('category').agg({
                        'impressions': 'sum',
                        'clicks': 'sum',
                        'query': 'nunique'
                    }).reset_index()
                    
                    category_summary.columns = ['Category', 'Impressions', 'Clicks', 'Unique Queries']
                    
                    # Calculate metrics
                    category_summary['CTR'] = (category_summary['Clicks'] / category_summary['Impressions'] * 100).round(2)
                    
                    # Create visualizations
                    st.subheader("Category Distribution")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(
                            category_summary,
                            values='Impressions',
                            names='Category',
                            title="Impressions by Query Category",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.pie(
                            category_summary,
                            values='Clicks',
                            names='Category',
                            title="Clicks by Query Category",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display category performance
                    st.subheader("Category Performance")
                    
                    fig = px.bar(
                        category_summary,
                        x='Category',
                        y=['CTR'],
                        title="CTR by Query Category",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Sample queries per category
                    st.subheader("Sample Queries by Category")
                    
                    for category in category_summary['Category']:
                        with st.expander(f"{category} Queries"):
                            sample_queries = classified_df[classified_df['category'] == category]
                            sample_queries = sample_queries.groupby('query').agg({
                                'impressions': 'sum',
                                'clicks': 'sum'
                            }).reset_index()
                            
                            sample_queries = sample_queries.sort_values('impressions', ascending=False).head(10)
                            st.dataframe(sample_queries)
    
    # Subtab 3: Multi-Dimensional Analysis
    with subtab3:
        st.subheader("Multi-Dimensional Analysis")
        st.caption("Analyze performance across multiple dimensions simultaneously.")
        
        # Dimension selectors
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_dim = st.selectbox("X-Axis Dimension", ["date", "country", "device", "search_type"])
        
        with col2:
            y_dim = st.selectbox("Y-Axis Metric", ["impressions", "clicks", "ctr", "position"])
        
        with col3:
            color_dim = st.selectbox("Color Dimension", ["None", "country", "device", "search_type"])
            color_dim = None if color_dim == "None" else color_dim
        
        # Create facet options
        facet_options = ["None", "country", "device", "search_type"]
        facet_col = st.selectbox("Facet By (Columns)", facet_options)
        facet_col = None if facet_col == "None" else facet_col
        
        # Aggregate data
        agg_dimensions = [x_dim]
        if color_dim and color_dim not in agg_dimensions:
            agg_dimensions.append(color_dim)
        if facet_col and facet_col not in agg_dimensions:
            agg_dimensions.append(facet_col)
        
        # Create a copy to avoid modifying the original
        multi_dim_df = df.copy()
        
        # Format date if selected
        if x_dim == 'date':
            multi_dim_df['date'] = pd.to_datetime(multi_dim_df['date'])
        
        # Group by dimensions
        grouped = multi_dim_df.groupby(agg_dimensions).agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'ctr': 'mean',
            'position': 'mean'
        }).reset_index()
        
        # Generate plot
        fig = px.line(
            grouped,
            x=x_dim,
            y=y_dim,
            color=color_dim,
            facet_col=facet_col,
            facet_col_wrap=2 if facet_col else None,
            title=f"{y_dim.title()} by {x_dim.title()}" + (f" and {color_dim.title()}" if color_dim else ""),
            height=600
        )
        
        # Customize layout based on dimensions
        if x_dim != 'date':
            fig.update_xaxes(type='category')
        
        if facet_col:
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
        st.plotly_chart(fig, use_container_width=True)

# Export functions
def generate_download_link(df, filename="gsc_data.csv", button_text="Download CSV"):
    """Generate a download link for the DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{button_text}</a>'
    return href

def export_section(df):
    """Create export section for data."""
    st.header("Export Data")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        export_format = st.selectbox("Export Format", ["CSV", "Excel"])
    
    with col2:
        filename = st.text_input("Filename", "gsc_data_export")
    
    if st.button("Generate Export"):
        if export_format == "CSV":
            download_link = generate_download_link(df, f"{filename}.csv", "Download CSV")
            st.markdown(download_link, unsafe_allow_html=True)
        else:
            # For Excel, we need to save to BytesIO
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='GSC Data')
            
            b64 = base64.b64encode(output.getvalue()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download Excel</a>'
            st.markdown(href, unsafe_allow_html=True)

# Scheduled Job Configuration
def setup_scheduled_jobs():
    """Configure scheduled data imports."""
    st.header("Scheduled Data Import")
    st.caption("Configure automatic imports of GSC data to BigQuery.")
    
    with st.expander("Schedule Configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            schedule_frequency = st.selectbox(
                "Update Frequency",
                ["Daily", "Weekly", "Monthly"]
            )
        
        with col2:
            if schedule_frequency == "Daily":
                schedule_time = st.time_input("Daily Update Time", datetime.strptime("02:00", "%H:%M").time())
            elif schedule_frequency == "Weekly":
                schedule_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                schedule_time = st.time_input("Time", datetime.strptime("02:00", "%H:%M").time())
            else:
                schedule_day = st.slider("Day of Month", 1, 28, 1)
                schedule_time = st.time_input("Time", datetime.strptime("02:00", "%H:%M").time())
        
        # Data retention settings
        data_retention = st.slider("Data Retention (days)", 30, 365, 90)
        
        # Query template
        query_template = f"""
        -- Example BigQuery Scheduled Query
        -- This would be scheduled to run {schedule_frequency.lower()} at {schedule_time}
        
        INSERT INTO `project.dataset.gsc_data`
        SELECT
          date,
          query,
          page,
          country,
          device,
          search_type,
          impressions,
          clicks,
          SAFE_DIVIDE(clicks, impressions) AS ctr,
          position
        FROM
          `project.dataset.searchanalyticsdata_*`
        WHERE
          _TABLE_SUFFIX BETWEEN
            FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 2 DAY)) AND
            FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY))
        """
        
        st.code(query_template, language="sql")
        
        # Scheduled query instructions
        st.markdown("""
        ### Setup Instructions
        1. Go to BigQuery Console
        2. Click "Create Query"
        3. Paste the above query (customize as needed)
        4. Click "Schedule"
        5. Set the schedule according to your chosen frequency
        6. Set the destination table
        7. Enable the schedule
        """)

# Main function
def main():
    # Setup authentication
    setup_authentication()
    
    # Create sidebar controls
    filters = create_sidebar()
    
    # Main content
    if 'df' not in st.session_state or filters['fetch_data']:
        if 'bq_client' in st.session_state:
            # Fetch data from BigQuery
            df = fetch_gsc_data(
                filters['project_id'],
                filters['dataset_id'],
                filters['table_id'],
                filters['start_date'],
                filters['end_date'],
                limit=filters['sample_size'] if filters['use_sample'] else None
            )
            
            if df is not None and not df.empty:
                # Process data in chunks if large
                df = process_data_chunks(df)
                
                # Save to session state
                st.session_state['df'] = df
                
                # Save to parquet for faster access
                save_to_parquet(df)
                
                st.success(f"✅ Data loaded successfully! {len(df):,} rows retrieved.")
            
        elif 'bq_client' not in st.session_state and filters['fetch_data']:
            st.error("Please set up BigQuery authentication first.")
    
    # Check if we can load from parquet if no data in session
    if 'df' not in st.session_state:
        df = load_from_parquet()
        if df is not None:
            st.session_state['df'] = df
            st.info("Loaded data from cache.")
    
    # Create main dashboard
    if 'df' in st.session_state:
        create_main_dashboard(st.session_state['df'], filters)
        
        # Add export section
        export_section(st.session_state['df'])
        
        # Add scheduled job configuration
        setup_scheduled_jobs()
    else:
        # Show instructions if no data
        st.info("""
        ### Welcome to the GSC BigQuery Analyzer!
        
        This app allows you to analyze your Google Search Console data stored in BigQuery.
        
        To get started:
        1. Set up BigQuery authentication in the sidebar
        2. Configure your project, dataset, and table IDs
        3. Select your date range and filters
        4. Click "Fetch Data"
        
        If you don't have GSC data in BigQuery yet, check out Google's documentation on how to [export Search Console data to BigQuery](https://developers.google.com/search/docs/data-tools/search-console-big-query-export).
        """)

# Add stylesheet
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px;
        color: #4A4A4A;
        border: 1px solid #E0E0E0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F0F2F6;
        border-color: #A0A0A0;
    }
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    # Ensure NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Run the app
    main()
