from flask import request, Response, json, jsonify, make_response
from singletons.db_conn import get_db_connection, release_db_connection
from singletons.gen_model import get_fallback_gen_model, get_gen_model
from singletons.logger import get_logger
from services.rag_llm import generate_with_llm
from services.vectorstore import retrieve_docs_from_query, retrieve_docs_from_product_code
import sqlite3
from collections import defaultdict

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

    try:
        _, product_details_list = retrieve_docs_from_query(
            conn, query, top_k=limit, skip=offset)
    finally:
        release_db_connection(conn)

    # Return the product_details_list, which is a list of dictionaries and JSON serializable
    return Response(json.dumps(product_details_list), mimetype='application/json')


def handle_get_co_occur_products():
    try:
        product_ids = request.args.get('product_ids')
        limit = int(request.args.get('limit', 5))

        # Debug logging to see what we're receiving
        logger.info(f"🔥 Received product_ids: {product_ids}")
        logger.info(f"🔥 Request args: {dict(request.args)}")

        if not product_ids:
            return Response(json.dumps({"message": "Missing product_id(s)"}), status=400, mimetype='application/json')

        db_path = 'modules/co-occur-products/co-occur-products.db'
        conn = None

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Prepare placeholders for SQL IN clause
            product_ids = product_ids.split(',')
            placeholders = ','.join('?' for _ in product_ids)
            query = f"""
            SELECT product_id, related_id
            FROM related_products
            WHERE product_id IN ({placeholders})
            ORDER BY product_id, related_id
        """
            logger.info(f"🔥 Executing query with product_ids: {product_ids}")
            cursor.execute(query, product_ids)
            rows = cursor.fetchall()

            # Aggregate related_ids for each product_id
            related_dict = defaultdict(list)
            for pid, rid in rows:
                related_dict[str(pid)].append(int(rid))

            # Limit the number of related_ids per product_id
            result = {}
            for pid in product_ids:
                related = related_dict.get(pid, [])[:limit]
                logger.info(
                    f"🔥 Finding co-occur products of product ID {pid} → {related}")
                result[pid] = related

            logger.info("🔥 Co-occur products result: %s", result)

            response = make_response(jsonify(result))
            response.headers["Cache-Control"] = "no-store"
            return response

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return Response(json.dumps({"message": "Database error occurred"}), status=500, mimetype='application/json')
        finally:
            if conn:
                conn.close()

    except ValueError as e:
        logger.error(f"Invalid parameter value: {e}")
        return Response(json.dumps({"message": "Invalid parameter value"}), status=400, mimetype='application/json')
    except Exception as e:
        logger.error(f"Unexpected error in handle_get_co_occur_products: {e}")
        return Response(json.dumps({"message": "An unexpected error occurred"}), status=500, mimetype='application/json')


def handle_get_related_products_semantics():
    product_code = request.args.get("product_code", 'NA')
    top_k = int(request.args.get("top_k", 5))
    logger.info(
        f"🔥 Received roduct code {product_code}")

    conn = get_db_connection()
    result = retrieve_docs_from_product_code(conn, product_code, top_k)
    if result is None:
        related_products = []
    else:
        _, related_products = result

    release_db_connection(conn)

    return Response(
        json.dumps(related_products, ensure_ascii=False),
        content_type='application/json'
    )
