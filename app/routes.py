from flask import Blueprint, render_template
from ml.eda import run_eda

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return "Flask berhasil jalan!"


@main.route('/eda')
def eda():
    eda_data = run_eda()
    return render_template(
        'eda.html',
        recency=eda_data['recency'],
        frequency=eda_data['frequency'],
        monetary=eda_data['monetary'],
        correlation=eda_data['correlation']
    )