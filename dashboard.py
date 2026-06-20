import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set page configuration
st.set_page_config(page_title="Customer Insights Dashboard", layout="wide")

# App Header
st.title(" Customer Insights Dashboard: CLV & Churn")
st.markdown("""
This dashboard analyzes customer behavior, identifies churn risks, and segments customers for targeted marketing.
""")

# Load Data (Cached for performance)
@st.cache_data
def load_and_process_data():
    data_path = "data/"
    prediction_date = pd.to_datetime("2025-06-18")

    customers = pd.read_csv(os.path.join(data_path, "customers.csv"))
    transactions = pd.read_csv(os.path.join(data_path, "transactions.csv"))
    churn_labels = pd.read_csv(os.path.join(data_path, "churn_labels.csv"))
    clv_labels = pd.read_csv(os.path.join(data_path, "clv_labels.csv"))
    events = pd.read_csv(os.path.join(data_path, "events.csv"))
    support = pd.read_csv(os.path.join(data_path, "support_tickets.csv"))

    # Cleaning
    customers['signup_date'] = pd.to_datetime(customers['signup_date'])
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])

    # Aggregations
    rfm = transactions.groupby('customer_id').agg(
        recency=('transaction_date', lambda x: (prediction_date - x.max()).days),
        frequency=('customer_id', 'count'),
        monetary=('amount', 'sum')
    )

    active_days = events.groupby('customer_id')['event_date'].nunique().rename('active_days_count')

    support_agg = support.groupby('customer_id').agg(
        ticket_count=('ticket_id', 'count'),
        avg_satisfaction=('satisfaction_score', 'mean')
    )

    # Master DF
    df = customers.merge(churn_labels, on='customer_id', how='left')
    df = df.merge(clv_labels, on='customer_id', how='left')
    df = df.merge(rfm, on='customer_id', how='left')
    df = df.merge(active_days, on='customer_id', how='left')
    df = df.merge(support_agg, on='customer_id', how='left').fillna(0)

    # Clustering
    scaler = StandardScaler()
    X = df[['recency', 'frequency', 'monetary', 'active_days_count']]
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    return df, support

df, support_raw = load_and_process_data()

# Sidebar Filters
st.sidebar.header("Filter Options")
selected_plan = st.sidebar.multiselect("Select Plan Type", options=df['plan_type'].unique(), default=df['plan_type'].unique())
df_filtered = df[df['plan_type'].isin(selected_plan)]

# Row 1: Key Metrics
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Customers", f"{len(df_filtered):,}")
col2.metric("Average CLV", f"${df_filtered['clv'].mean():,.2f}")
col3.metric("Churn Rate", f"{df_filtered['churn_30d'].mean()*100:.1f}%")
col4.metric("Avg Monetary", f"${df_filtered['monetary'].mean():,.2f}")
col5.metric("Avg Satisfaction", f"{df_filtered[df_filtered['avg_satisfaction'] > 0]['avg_satisfaction'].mean():.1f}/5")

# Row 2: Visualizations
st.markdown("---")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Distribution of Customer Lifetime Value")
    fig, ax = plt.subplots()
    sns.histplot(df_filtered['clv'], bins=30, kde=True, color='teal', ax=ax)
    ax.set_title("CLV Distribution")
    st.pyplot(fig)

with col_b:
    st.subheader("Customer Segments (CLV vs Churn)")
    segment_analysis = df_filtered.groupby('cluster').agg({
        'clv': 'mean',
        'churn_30d': 'mean',
        'ticket_count': 'mean',
        'customer_id': 'count'
    }).rename(columns={'customer_id': 'count', 'churn_30d': 'churn_rate'})

    fig, ax = plt.subplots()
    sns.scatterplot(data=segment_analysis, x='clv', y='churn_rate', size='count', hue=segment_analysis.index,
                    palette='viridis', sizes=(100, 1000), ax=ax)
    ax.set_xlabel("Average CLV")
    ax.set_ylabel("Churn Rate")
    st.pyplot(fig)

# Row 3: Support Analysis
st.markdown("---")
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Support Ticket Categories")
    # Filter raw support data to only include customers in filtered df
    filtered_support = support_raw[support_raw['customer_id'].isin(df_filtered['customer_id'])]
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_support, y='ticket_category', order=filtered_support['ticket_category'].value_counts().index, palette='magma', ax=ax)
    st.pyplot(fig)

with col_d:
    st.subheader("Segment Deep Dive")
    st.dataframe(segment_analysis.style.format({'clv': '${:,.2f}', 'churn_rate': '{:.2%}', 'ticket_count': '{:.1f}'}))

st.markdown("""
### 💡 Strategic Recommendations
1. **Cluster 3 (High Value/Low Churn)**: Focus on 'VIP' loyalty programs and referral incentives.
2. **Cluster 0 (High Churn Risk)**: Immediate outreach via personalized retention emails and discount offers.
3. **Support Optimization**: Address the most frequent ticket categories (e.g., Technical or Billing) to improve overall satisfaction and reduce churn friction.
""")
