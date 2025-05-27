from singletons.logger import get_logger
from singletons.embedding_model import embed_text
from singletons.db_conn import release_db_connection
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = get_logger()


COLLECTION_NAME = "product_pg_embeddings"
SCHEMA_NAME = "public"


def retrieve_docs_from_query(conn, query, top_k=3, skip=0):
    """Retrieve documents and product details using vector distance."""
    try:
        query_embedding = embed_text(query)
        logger.info(
            f"Query: '{query}', Embedding (first 5 dims): {query_embedding[:5]}")

        if all(v == 0.0 for v in query_embedding):
            logger.warning(
                "Query embedding failed (all zeros), search results will be inaccurate")

        cursor = conn.cursor()
        cursor.execute(f"SET search_path TO {SCHEMA_NAME}, public;")

        sql_query = """
        SELECT
            e.document,
            e.cmetadata,
            e.embedding <-> %s::vector AS distance,
            p.product_id,
            p.product_total_ratings,
            p.product_overall_stars,
            p.product_total_orders,
			p.product_discount_percentage,
			p.product_unit_price,
			p.category_id,
            pi.product_image_url
        FROM
            langchain_pg_collection c
        JOIN
            langchain_pg_embedding e ON c.id = e.collection_id
        LEFT JOIN
            products p ON p.product_code = e.cmetadata->>'product_code'
        LEFT JOIN LATERAL
            (SELECT product_image_url
            FROM product_images
            WHERE product_images.product_id = p.product_id
            LIMIT 1) pi ON TRUE
        WHERE
			c.name = %s
            AND (p.product_stock_quantity > 0 OR p.product_stock_quantity IS NULL)
        ORDER BY
            distance ASC
        LIMIT %s
        OFFSET %s;
        """
        logger.info(
            f"Executing vector search with COLLECTION_NAME: {COLLECTION_NAME}, top_k: {top_k}")
        cursor.execute(sql_query, (query_embedding,
                       COLLECTION_NAME, top_k, skip))

        fetched_rows = cursor.fetchall()
        logger.info(
            f"Number of rows returned by vector search: {len(fetched_rows)}")

        documents_with_distance = []
        product_details_list = []

        for row_idx, row_data in enumerate(fetched_rows):
            document_text = row_data[0]
            metadata_json = row_data[1]
            distance = row_data[2]

            logger.info(
                f"Row {row_idx} - Document text (first 100 chars): {document_text[:100]}...")
            logger.info(f"Row {row_idx} - Raw Metadata JSON: {metadata_json}")
            logger.info(f"Row {row_idx} - Distance: {distance}")

            # Parse metadata
            if isinstance(metadata_json, dict):
                metadata = metadata_json
            elif metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    logger.error(
                        f"Row {row_idx} - Failed to parse metadata JSON: {metadata_json}")
                    metadata = {}
            else:
                metadata = {}

            logger.info(f"Row {row_idx} - Parsed Metadata: {metadata}")

            # Create document for LLM
            document = Document(page_content=document_text, metadata=metadata)
            documents_with_distance.append((document, float(distance)))

            # Extract product details for UI
            product_details = {
                "product_code": metadata.get("product_code", "Unknown"),
                "product_name": metadata.get("product_name", "Unknown"),
                "category_name": metadata.get("category_name", "Unknown"),
                "description": document_text,  # Use document text as description
                "product_id": row_data[3],
                "total_ratings": row_data[4],
                "overall_stars": row_data[5],
                "total_orders": row_data[6],
                "discount_percentage": row_data[7],
                # JSON string of sizes and prices, e.g. {"product_sizes": "S|M|L", "product_prices": "89000|94000|99000"}
                "sizes_prices": row_data[8],
                "category_id": row_data[9],
                "image_url": row_data[10],
            }
            product_details_list.append(product_details)

        cursor.close()
        return documents_with_distance, product_details_list
    except Exception as e:
        logger.error(f"Error in retrieve_docs: {e}", exc_info=True)
        return [], []


def retrieve_docs_from_product_code(conn, product_code, top_k=5):
    """Retrieve a single document by product code and find similar products."""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SET search_path TO {SCHEMA_NAME}, public;")

        # Get the document and metadata for the given product_code
        sql_query = """
		SELECT
			e.document,
			e.cmetadata
		FROM
			langchain_pg_collection c
		JOIN
			langchain_pg_embedding e ON c.id = e.collection_id
		WHERE
			c.name = %s AND e.cmetadata->>'product_code' = %s;
		"""
        logger.info(
            f"Executing document retrieval for product_code: {product_code}")
        cursor.execute(sql_query, (COLLECTION_NAME, product_code))

        query_product_row = cursor.fetchone()
        documents = []
        if query_product_row:
            document_text = query_product_row[0]
            metadata_json = query_product_row[1]

            # Parse metadata
            if isinstance(metadata_json, dict):
                metadata = metadata_json
            elif metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to parse metadata JSON: {metadata_json}")
                    metadata = {}
            else:
                metadata = {}

            embedding = embed_text(document_text)

            # Find similar products (excluding itself)
            similar_sql = """
			SELECT
				p.product_id,
				p.product_name,
				p.category_id,
				p.product_unit_price,
				p.product_discount_percentage,
				pi.product_image_url,
				e.document,
				e.cmetadata
			FROM
				langchain_pg_collection c
			JOIN
				langchain_pg_embedding e ON c.id = e.collection_id
			LEFT JOIN
				products p ON p.product_code = e.cmetadata->>'product_code'
            LEFT JOIN LATERAL
                (SELECT product_image_url
                FROM product_images
                WHERE product_images.product_id = p.product_id
                LIMIT 1) pi ON TRUE
			WHERE
				c.name = %s
				AND e.cmetadata->>'product_code' != %s
				AND (p.product_stock_quantity > 0 OR p.product_stock_quantity IS NULL)
			ORDER BY
				e.embedding <=> %s::vector ASC
			LIMIT %s
			"""
            cursor.execute(similar_sql, (COLLECTION_NAME,
                           product_code, embedding, top_k))

            related_product_rows = cursor.fetchall()
            related_product_details = []
            for row in related_product_rows:
                # Parse sizes and prices
                sizes_prices_obj = row[3]
                sizes, prices = parse_product_sizes_prices(sizes_prices_obj)
                related_product_details.append({
                    "product_id": row[0],
                    "product_name": row[1],
                    "category_id": row[2],
                    "sizes": sizes,
                    "prices": prices,
                    "discount_percentage": row[4],
                    "image_url": row[5],
                })
                # Create langchain document
                documents.append(Document(
                    page_content=row[6], metadata=row[7]))

            cursor.close()
            logger.debug("Retrieved product details: %s",
                         related_product_details)
            return documents, related_product_details

        else:
            logger.warning(
                f"No document found for product_code: {product_code}")
            cursor.close()
            return None
    except Exception as e:
        logger.error(
            f"Error in retrieve_docs_from_product_code: {e}", exc_info=True)
        return None


# Util function (de day tam)
def parse_product_sizes_prices(sizes_prices_dict):
    """Parse sizes and prices from a JSON object."""
    if not sizes_prices_dict:
        return {"sizes": [], "product_prices": []}

    try:
        sizes = sizes_prices_dict.get("product_sizes", "").split("|")
        prices = sizes_prices_dict.get("product_prices", "")
        prices = list(map(int, str(prices).split("|")))
        logger.debug("Parsed sizes: %s, prices: %s", sizes, prices)
        return sizes, prices
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse sizes and prices: {e}")
        return {"product_sizes": [], "product_prices": []}
