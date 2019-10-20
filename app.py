from flask import Flask
# from schema.schemaBp import schema_bp
# from keywordSearch.keywordSearchBp import keyword_search_bp
from multiview.multiviewBp import multiview_bp

app = Flask(__name__)
# app.register_blueprint(schema_bp, url_prefix='/schema')
# app.register_blueprint(keyword_search_bp, url_prefix='/keywordsearch')
app.register_blueprint(multiview_bp, url_prefix='/multiview')

@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run()