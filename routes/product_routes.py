from flask import Blueprint, jsonify
from handlers.product_handlers import handle_search_rag, handle_search_semantics, handle_get_co_occur_products, handle_get_related_products_semantics

product_bp = Blueprint('products', __name__, url_prefix='/product')


@product_bp.route('/search/rag', methods=['GET'])
def handle_rag_route():
    return handle_search_rag()


@product_bp.route('/search/semantics', methods=['GET'])
def handle_semantics_route():
    return handle_search_semantics()


@product_bp.route('/related/co-occur', methods=['GET'])
def handle_co_occur_route():
    return handle_get_co_occur_products()


@product_bp.route('/related/semantics', methods=['GET'])
def handle_related_semantics_route():
    return handle_get_related_products_semantics()
