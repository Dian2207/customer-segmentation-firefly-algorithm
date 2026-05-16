from flask import Blueprint, render_template

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return "Flask berhasil jalan!"

@main.route('/eda')
def eda():
    return render_template('eda.html')