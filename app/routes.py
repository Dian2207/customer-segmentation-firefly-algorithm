from flask import Blueprint, render_template
from ml.eda import run_eda

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