import pandas as pd
import numpy as np


def run_eda():

    print("===================================")
    print("STEP 4 : EXPLORATORY DATA ANALYSIS")
    print("===================================")

    rfm = pd.read_csv(
        "data/processed/rfm_raw.csv"
    )

    print("\nData RFM berhasil dibaca")
    print(rfm.head())

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

    monetary_labels = [

        "0-200",
        "201-600",
        "601-1000",
        "1001-3000",   
        "3001-5000",
        "5001-7000",
        "7001-10000",
        "10001-15000",
        "15000+"

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
            (rfm['Monetary'] <= 3000)
        ]
    ),

    len(
        rfm[
            (rfm['Monetary'] >= 3001) &
            (rfm['Monetary'] <= 5000)
        ]
    ),

    len(
        rfm[
            (rfm['Monetary'] >= 5001) &
            (rfm['Monetary'] <= 7000)
        ]
    ),

    len(
        rfm[
            (rfm['Monetary'] >= 7001) &
            (rfm['Monetary'] <= 10000)
        ]
    ),

    len(
        rfm[
            (rfm['Monetary'] >= 10001) &
            (rfm['Monetary'] <= 15000)
        ]
    ),

    len(
        rfm[
            rfm['Monetary'] > 15000
        ]
    )

]

    correlation = rfm[
        ['Recency', 'Frequency', 'Monetary']
    ].corr().round(2)

    print("\nCorrelation Matrix")
    print(correlation)

    total_customer = len(rfm)

    avg_recency = round(
        rfm['Recency'].mean()
    )

    avg_frequency = round(
        rfm['Frequency'].mean()
    )

    avg_monetary = round(
        rfm['Monetary'].mean()
    )

    print("\n===================================")
    print("EDA SELESAI")
    print("===================================")

    return {

    "total_customer":
    f"{total_customer:,}".replace(",", "."),

    "avg_recency":
    f"{avg_recency} Hari",

    "avg_frequency":
    f"{avg_frequency} Transaksi",

    "avg_monetary":
    f"Rp {avg_monetary:,}".replace(",", "."),

    "recency_labels":
    recency_labels,

    "recency_counts":
    recency_counts.tolist(),

    "frequency_labels":
    frequency_labels,

    "frequency_counts":
    frequency_counts,

    "monetary_labels":
    monetary_labels,

    "monetary_counts":
    monetary_counts,

    "correlation":
    correlation.values.tolist()

}