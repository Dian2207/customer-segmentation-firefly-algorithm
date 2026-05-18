import pandas as pd
from flask import Blueprint, render_template
from ml.eda import run_eda
from ml.segmentasi_visual import get_cluster_visual_data


main = Blueprint('main', __name__)

@main.route('/')
def home():
    return "Flask berhasil jalan!"


@main.route('/eda')
def eda():
    eda_result = run_eda()
    return render_template(
        'eda.html',
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

    df_summary = pd.read_csv("data/processed/cluster_summary.csv")

    return render_template(
        "segmentasi.html",
        recency=data["recency"],
        frequency=data["frequency"],
        monetary=data["monetary"],
        cluster=data["cluster"],

        cluster_summary=df_summary.to_dict(orient="records")
    )