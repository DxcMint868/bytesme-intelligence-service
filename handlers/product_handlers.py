from flask import request, Response, json
from singletons.db_conn import get_db_connection
from singletons.gen_model import get_fallback_gen_model, get_gen_model
from singletons.logger import get_logger
from services.rag_llm import generate_with_llm
from services.vectorstore import retrieve_docs

logger = get_logger()


def handle_search_rag():
    query = request.args.get('query')
    if not query:
        return Response(json.dumps({"message": "Query parameter is missing"}), status=400, mimetype='application/json')

    # Prepared injection dependencies
    gen_model = get_gen_model()

    def on_rate_limit(e):
        logger.warning(f"Rate limit exceeded: {e}")
        fallback_gen_model = get_fallback_gen_model()
        # Try again with fallback model
        generate_with_llm(query, fallback_gen_model)

    def on_unauthorized(e):
        logger.error(f"Unauthorized access: {e}")
        return Response(json.dumps({"message": "Unauthorized access"}), status=401, mimetype='application/json')

    def on_generic_error(e):
        logger.error(f"Generic error occurred: {e}")
        return Response(json.dumps({"message": "An error occurred"}), status=500, mimetype='application/json')

    return Response(
        generate_with_llm(
            query,
            gen_model,
            on_rate_limit=on_rate_limit,
            on_unauthorized=on_unauthorized,
            on_generic_error=on_generic_error
        ),
        mimetype='text/event-stream')


def handle_search_semantics():
    query = request.args.get('query', 'Đồ ăn và nước uống')
    limit = int(request.args.get('limit', 10))
    offset = int(request.args.get('offset', 0))

    logger.info(f'Query: {query}')
    conn = get_db_connection()

    _, product_details_list = retrieve_docs(
        conn, query, top_k=limit, skip=offset)

    # Return the product_details_list, which is a list of dictionaries and JSON serializable
    return Response(json.dumps(product_details_list), mimetype='application/json')
