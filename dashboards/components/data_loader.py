"""
Data Loader

Centralized data loading with Streamlit caching.
"""

import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict


# Data directory path
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


@st.cache_data
def load_transactions() -> pd.DataFrame:
    """
    Load cleaned transaction data with date parsing.

    Returns:
        DataFrame with 354,321 transactions
        Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate,
                 UnitPrice, CustomerID, Country, TotalPrice, DayOfWeek, HourOfDay
    """
    df = pd.read_csv(
        DATA_DIR / "cleaned_uk_data.csv",
        parse_dates=["InvoiceDate"]
    )
    return df


@st.cache_data
def load_rfm() -> pd.DataFrame:
    """
    Load RFM data.

    Returns:
        DataFrame with 3,920 customers
        Columns: CustomerID, Recency, Frequency, Monetary
    """
    df = pd.read_csv(DATA_DIR / "rfm_data.csv")
    return df


@st.cache_data
def load_customer_features(scaled: bool = False) -> pd.DataFrame:
    """
    Load customer features (original or scaled).

    Args:
        scaled: If True, load standardized features; if False, load original

    Returns:
        DataFrame with 3,920 customers, 16 feature columns
    """
    filename = "customer_features_scaled.csv" if scaled else "customer_features.csv"
    df = pd.read_csv(DATA_DIR / filename)
    return df


@st.cache_data
def load_clusters(k: int = 3) -> pd.DataFrame:
    """
    Load cluster assignments for specified k.

    Args:
        k: Number of clusters (3 or 4)

    Returns:
        DataFrame with CustomerID and Cluster columns
    """
    df = pd.read_csv(DATA_DIR / f"customer_clusters_k{k}.csv")
    return df


@st.cache_data
def load_merged_data(k: int = 3) -> pd.DataFrame:
    """
    Load transactions merged with cluster assignments.

    Args:
        k: Number of clusters (3 or 4)

    Returns:
        Transaction DataFrame with Cluster column added
    """
    transactions = load_transactions()
    clusters = load_clusters(k)

    # Merge on CustomerID
    merged = transactions.merge(clusters, on="CustomerID", how="left")
    return merged


def get_date_range() -> Tuple[datetime, datetime]:
    """
    Get min/max dates from transaction data.

    Returns:
        Tuple of (min_date, max_date)
    """
    transactions = load_transactions()
    min_date = transactions["InvoiceDate"].min().to_pydatetime()
    max_date = transactions["InvoiceDate"].max().to_pydatetime()
    return min_date, max_date


def check_data_files() -> Dict[str, bool]:
    """
    Check existence of all required data files.

    Returns:
        Dict with file names as keys, existence status as values
    """
    required_files = [
        "cleaned_uk_data.csv",
        "rfm_data.csv",
        "customer_features.csv",
        "customer_features_scaled.csv",
        "customer_clusters_k3.csv",
        "customer_clusters_k4.csv",
    ]

    return {f: (DATA_DIR / f).exists() for f in required_files}
