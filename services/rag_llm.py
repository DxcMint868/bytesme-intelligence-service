from singletons.logger import get_logger
from singletons.db_conn import get_db_connection, release_db_connection
from services.vectorstore import retrieve_docs_from_query
from langchain.prompts import PromptTemplate
import requests
import json

logger = get_logger()


# Define RAG prompt
PROMPT_TEMPLATE = """
You are an AI assistant for an F&B app. Use the following product information to answer the user's query. Provide concise, relevant responses.
Context: {context}
Query: {question}
Answer:
"""


def generate_with_llm(
        query,
        llm,
        on_rate_limit=None,
        on_unauthorized=None,
        on_generic_error=None
):
    try:
        conn = get_db_connection()
        # Get documents and product details from vector search
        retrieved_docs_with_distance, product_details_list = retrieve_docs_from_query(
            conn, query, top_k=5)
        # Stream document to front-end to simulate real thinking
        content = ""
        for idx, (doc, distance) in enumerate(retrieved_docs_with_distance):
            content = doc.page_content[:200] + "..." if len(
                doc.page_content) > 200 else doc.page_content
            content += "Đánh giá mức độ trùng khớp so với tìm kiếm người dùng: " + \
                str(round(1 - distance, 4)) + "\n\n"
        yield f"data: {json.dumps({'type': 'thinking', 'chunk': content})}\n\n"
        # Prepare context for LLM
        if retrieved_docs_with_distance:
            context_parts = [doc.page_content for doc,
                             dist in retrieved_docs_with_distance]
            context_str = "\n\n---\n\n".join(context_parts)
            logger.info(
                f"Context for LLM (first 300 chars): {context_str[:300]}...")
        else:
            context_str = "No specific product information was found related to your query."
            logger.info(
                "No documents found by vector search to form context.")

        # Format prompt
        prompt = PromptTemplate(template=PROMPT_TEMPLATE,
                                input_variables=["context", "question"])
        full_prompt_text = prompt.format(
            context=context_str, question=query)
        logger.info(
            f"Full prompt for LLM (first 300 chars): {full_prompt_text[:300]}...")
        # Stream AI response
        llmOutput = ''
        for chunk in llm.stream(full_prompt_text):
            if isinstance(chunk, str):
                text = chunk
                llmOutput += text
            elif hasattr(chunk, "content"):
                text = chunk.content
                llmOutput += text
            else:
                text = str(chunk)
                llmOutput += text
            yield f"data: {json.dumps({'type': 'answer', 'chunk': text})}\n\n"
        # Stream product details
        for product in product_details_list:
            yield f"data: {json.dumps({'type': 'product', 'data': product})}\n\n"
        # Signal end of stream
        yield "data: [DONE]\n\n"
        logger.info("All streaming finished successfully")
        logger.info("LLM streaming finished successfully")
        logger.info(f"Full LLM output: {llmOutput}...")
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        logger.error(
            f"HTTP error during streaming query: {e}, Status Code: {status_code}", exc_info=True)
        if status_code == 429:
            if on_rate_limit:
                on_rate_limit(e)
        elif status_code == 401:
            if on_unauthorized:
                on_unauthorized(e)
        elif status_code == 500:
            if on_generic_error:
                on_generic_error(e)
        # yield f"data: {json.dumps({'type': 'answer', 'chunk': error_message, 'error': True})}\n\n"
        yield "data: [DONE]\n\n"

    finally:
        if conn:
            release_db_connection(conn)
