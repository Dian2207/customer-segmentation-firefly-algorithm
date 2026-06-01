import pandas as pd
import numpy as np
from flask import Blueprint, render_template
from ml.eda import run_eda
from ml.segmentasi_visual import get_cluster_visual_data


main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('dashboard.html')


@main.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@main.route('/eda')
def eda():
    eda_result = run_eda()
    return render_template(
        'eda.html',
        total_customer=
        eda_result['total_customer'],
        avg_recency=
        eda_result['avg_recency'],
        avg_frequency=
        eda_result['avg_frequency'],
        avg_monetary=
        eda_result['avg_monetary'],
        recency_labels=
        eda_result['recency_labels'],
        recency_counts=
        eda_result['recency_counts'],
        frequency_labels=
        eda_result['frequency_labels'],
        frequency_counts=
        eda_result['frequency_counts'],
        monetary_labels=
        eda_result['monetary_labels'],
        monetary_counts=
        eda_result['monetary_counts'],
        correlation=
        eda_result['correlation']

    )

@main.route("/segmentasi")
def segmentasi():

    data = get_cluster_visual_data()

    df_summary = pd.read_csv("data/result/cluster_summary.csv")

    return render_template(
        "segmentasi.html",
        recency=data["recency"],
        frequency=data["frequency"],
        monetary=data["monetary"],
        cluster=data["cluster"],

        cluster_summary=df_summary.to_dict(orient="records")
    )

@main.route('/stability')
def stability():
    df = pd.read_csv(
        "data/result/parameter_stability.csv"
    )
    iterations = df[
        'iteration'

    ].tolist()
    kmeans = df[
        'kmeans_silhouette'
    ].tolist()
    firefly = df[
        'firefly_silhouette'
    ].tolist()
    hybrid = df[
        'hybrid_firefly_silhouette'
    ].tolist()
    hybrid_ga = df[
        'hybrid_ga_silhouette'
    ].tolist()
    kmeans_time = df[
        'kmeans_time'
    ].tolist()
    firefly_time = df[
        'firefly_time'
    ].tolist()
    hybrid_time = df[
        'hybrid_firefly_time'
    ].tolist()
    hybrid_ga_time = df[
        'hybrid_ga_time'
    ].tolist()
    kmeans_avg = round(
        np.mean(kmeans),
        4
    )

    firefly_avg = round(
        np.mean(firefly),
        4
    )
    hybrid_avg = round(
        np.mean(hybrid),
        4
    )
    hybrid_ga_avg = round(
        np.mean(hybrid_ga),
        4
    )
    kmeans_avg_time = round(
        np.mean(kmeans_time),
        4
    )
    firefly_avg_time = round(
        np.mean(firefly_time),
        4
    )
    hybrid_avg_time = round(
        np.mean(hybrid_time),
        4
    )
    hybrid_ga_avg_time = round(
        np.mean(hybrid_ga_time),
        4
    )

    silhouette_dict = {
        "K-Means":
        kmeans_avg,
        "Firefly":
        firefly_avg,
        "Hybrid KMeans + Firefly":
        hybrid_avg,
        "Hybrid KMeans + GA":
        hybrid_ga_avg
    }

    best_method = max(
        silhouette_dict,
        key=silhouette_dict.get
    )

    best_silhouette = silhouette_dict[
        best_method
    ]

    time_dict = {
        "K-Means":
        kmeans_avg_time,
        "Firefly":
        firefly_avg_time,
        "Hybrid KMeans + Firefly":
        hybrid_avg_time,
        "Hybrid KMeans + GA":
        hybrid_ga_avg_time
    }

    fastest = min(
        time_dict,
        key=time_dict.get
    )

    stability_dict = {
        "K-Means":
        np.std(kmeans),
        "Firefly":
        np.std(firefly),
        "Hybrid KMeans + Firefly":
        np.std(hybrid),
        "Hybrid KMeans + GA":
        np.std(hybrid_ga)

    }

    stable = min(
        stability_dict,
        key=stability_dict.get
    )

    total_iteration = len(
        iterations
    )

    analysis = f"""
    Berdasarkan hasil pengujian sebanyak
    {total_iteration} iterasi,
    metode dengan silhouette score terbaik
    adalah {best_method}
    dengan rata-rata silhouette score
    sebesar {best_silhouette}.

    Dari sisi execution time,
    metode tercepat adalah {fastest}
    karena memiliki rata-rata waktu
    komputasi paling rendah.

    Berdasarkan standar deviasi
    silhouette score,
    metode paling stabil adalah
    {stable}
    karena memiliki fluktuasi
    nilai paling kecil.

    Secara keseluruhan,
    metode {best_method}
    menunjukkan performa terbaik
    untuk customer segmentation.
    """
    
    return render_template(
        "stability.html",
        iterations=iterations,
        kmeans=kmeans,
        firefly=firefly,
        hybrid=hybrid,
        hybrid_ga=hybrid_ga,
        kmeans_time=kmeans_time,
        firefly_time=firefly_time,
        hybrid_time=hybrid_time,
        hybrid_ga_time=hybrid_ga_time,
        total_iteration=total_iteration,
        best_silhouette=best_silhouette,
        fastest_method=fastest,
        analysis=analysis,
        best_method=best_method,
        fastest=fastest,
        stable=stable
    )