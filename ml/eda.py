# =========================================
# EDA CUSTOMER SEGMENTATION
# =========================================

import pandas as pd


def run_eda():

    print("===================================")
    print("STEP 4 : EXPLORATORY DATA ANALYSIS")
    print("===================================")

    # =========================================
    # LOAD DATA RFM
    # =========================================
    rfm = pd.read_csv(
        "data/processed/rfm.csv"
    )

    print("\nData RFM berhasil dibaca")
    print(rfm.head())

    # =========================================
    # INFO DATA
    # =========================================
    print("\nINFO DATA")
    print(rfm.info())

    # =========================================
    # DESKRIPSI STATISTIK
    # =========================================
    print("\nDESKRIPSI STATISTIK")
    print(rfm.describe())

    # =========================================
    # CEK NULL VALUE
    # =========================================
    print("\nNULL VALUE")
    print(rfm.isnull().sum())

    # =========================================
    # CORRELATION MATRIX
    # =========================================
    correlation = rfm.corr()

    print("\nCORRELATION MATRIX")
    print(correlation)

    print("\n===================================")
    print("EDA SELESAI")
    print("===================================")

    # =========================================
    # RETURN DATA KE FLASK
    # =========================================
    return {

        "recency":
        rfm['Recency'].tolist(),

        "frequency":
        rfm['Frequency'].tolist(),

        "monetary":
        rfm['Monetary'].tolist(),

        "correlation":
        correlation.values.tolist()
    }