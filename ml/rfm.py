import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

def create_rfm(df):
    print("STEP 2: Membuat RFM")

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate'])

    df['TotalPrice'] = df['Quantity'] * df['Price']

    NOW = dt.datetime(2011,12,10)

    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (NOW - x.max()).days,
        'Invoice': 'count',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency','Frequency','Monetary']

    rfm = rfm.head(100)

    scaler = MinMaxScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    rfm_scaled = pd.DataFrame(
        rfm_scaled,
        columns=['Recency','Frequency','Monetary']
    )

    rfm_scaled.to_csv("data/processed/rfm.csv", index=False)

    print("rfm.csv berhasil dibuat")
    return rfm_scaled