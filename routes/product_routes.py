from flask import Blueprint, jsonify
from handlers.product_handlers import handle_search_rag, handle_search_semantics

product_bp = Blueprint('products', __name__, url_prefix='/product')


@product_bp.route('/search/rag', methods=['GET'])
def handle_rag_route():
    return handle_search_rag()


@product_bp.route('/search/semantics', methods=['GET'])
def handle_semantics_route():
    return handle_search_semantics()
