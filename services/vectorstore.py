from singletons.logger import get_logger
from singletons.embedding_model import get_embedding_model
from singletons.db_conn import release_db_connection
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = get_logger()


EMBEDDING_DIMENSIONS = 768
COLLECTION_NAME = "product_pg_embeddings"
SCHEMA_NAME = "public"


def embed_text(text):
    """Wrapper function to safely create embeddings"""
    try:
        model = get_embedding_model()
        embeddings = model.encode(text, normalize_embeddings=True)
        logger.info(
            f"Generated embeddings for text: {text[:50]}... with shape {embeddings.shape}")
        # Log first 5 values for debugging
        logger.info(f"Embedding: {embeddings[:5]}...")
        return embeddings.tolist()
    except Exception as e:
        print(f"Error embedding text: {e}")
        # Return zeros array as fallback (with proper dimension)
        return [0.0] * EMBEDDING_DIMENSIONS


def retrieve_docs(conn, query, top_k=3, skip=0):
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
