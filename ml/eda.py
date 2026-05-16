import pandas as pd
import numpy as np


def run_eda():

    print("===================================")
    print("STEP 4 : EXPLORATORY DATA ANALYSIS")
    print("===================================")

    # =========================================
    # LOAD DATA
    # =========================================

    rfm = pd.read_csv(
        "data/processed/rfm_raw.csv"
    )

    print("\nData RFM berhasil dibaca")
    print(rfm.head())

    # =========================================
    # RECENCY HISTOGRAM
    # =========================================

    recency_bins = [

        0,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        np.inf

    ]

    recency_counts, _ = np.histogram(

        rfm['Recency'],

        bins=recency_bins

    )

    # =========================================
    # LABEL
    # =========================================

    recency_labels = [

        "0",
        "100",
        "200",
        "300",
        "400",
        "500",
        "600",
        "700",

    ]
    # =========================================
    # FREQUENCY HISTOGRAM
    # =========================================

    frequency_labels = [

        "1",
        "2-3",
        "4-5",
        "6-10",
        "11-20",
        "21-50",
        "51-100",
        "100+"

    ]

    frequency_counts = [

        len(rfm[rfm['Frequency'] == 1]),

        len(
            rfm[
                (rfm['Frequency'] >= 2) &
                (rfm['Frequency'] <= 3)
            ]
        ),

        len(
            rfm[
                (rfm['Frequency'] >= 4) &
                (rfm['Frequency'] <= 5)
            ]
        ),

        len(
            rfm[
                (rfm['Frequency'] >= 6) &
                (rfm['Frequency'] <= 10)
            ]
        ),

        len(
            rfm[
                (rfm['Frequency'] >= 11) &
                (rfm['Frequency'] <= 20)
            ]
        ),

        len(
            rfm[
                (rfm['Frequency'] >= 21) &
                (rfm['Frequency'] <= 50)
            ]
        ),

        len(
            rfm[
                (rfm['Frequency'] >= 51) &
                (rfm['Frequency'] <= 100)
            ]
        ),

        len(
            rfm[
                rfm['Frequency'] > 100
            ]
        )

    ]

    # =========================================
    # MONETARY HISTOGRAM
    # =========================================

    monetary_labels = [

        "0-200",
        "201-600",
        "601-1000",
        "1001-1400",
        "1401-1800",
        "1801-2200",
        "2201-2600",
        "2601-3000",
        "3000+"

    ]

    monetary_counts = [

        len(
            rfm[
                (rfm['Monetary'] >= 0) &
                (rfm['Monetary'] <= 200)
            ]
        ),

        len(
            rfm[
                (rfm['Monetary'] >= 201) &
                (rfm['Monetary'] <= 600)
            ]
        ),

        len(
            rfm[
                (rfm['Monetary'] >= 601) &
                (rfm['Monetary'] <= 1000)
            ]
        ),

        len(
            rfm[
                (rfm['Monetary'] >= 1001) &
                (rfm['Monetary'] <= 1400)
            ]
        ),

        len(
            rfm[
                (rfm['Monetary'] >= 1401) &
                (rfm['Monetary'] <= 1800)
            ]
        ),

        len(
            rfm[
                (rfm['Monetary'] >= 1801) &
                (rfm['Monetary'] <= 2200)
            ]
        ),

        len(
            rfm[
                (rfm['Monetary'] >= 2201) &
                (rfm['Monetary'] <= 2600)
            ]
        ),

        len(
            rfm[
                (rfm['Monetary'] >= 2601) &
                (rfm['Monetary'] <= 3000)
            ]
        ),

        len(
            rfm[
                rfm['Monetary'] > 3000
            ]
        )

    ]

    # =========================================
    # CORRELATION MATRIX
    # =========================================

    correlation = rfm[
        ['Recency', 'Frequency', 'Monetary']
    ].corr().round(2)

    print("\nCorrelation Matrix")
    print(correlation)

    print("\n===================================")
    print("EDA SELESAI")
    print("===================================")

    # =========================================
    # RETURN DATA
    # =========================================

    return {

        # =====================================
        # RECENCY
        # =====================================

        "recency_labels":
        recency_labels,

        "recency_counts":
        recency_counts.tolist(),

        # =====================================
        # FREQUENCY
        # =====================================

        "frequency_labels":
        frequency_labels,

        "frequency_counts":
        frequency_counts,

        # =====================================
        # MONETARY
        # =====================================

        "monetary_labels":
        monetary_labels,

        "monetary_counts":
        monetary_counts,

        # =====================================
        # CORRELATION
        # =====================================

        "correlation":
        correlation.values.tolist()

    }