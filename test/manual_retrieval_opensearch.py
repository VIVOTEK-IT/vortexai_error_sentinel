from opensearchpy import OpenSearch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from error_log_monitor.config import load_config
import logging

logger = logging.getLogger(__name__)


# Connect to OpenSearch
def connect_opensearch():
    """Connect to OpenSearch."""
    config = load_config().opensearch
    try:
        client = OpenSearch(
            hosts=[
                {
                    'host': config.host,
                    'port': config.port,
                }
            ],
            http_auth=(config.username, config.password) if config.username else None,
            use_ssl=True,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        logger.info(f"Connected to OpenSearch at {config.host}:{config.port}")
    except Exception as e:
        logger.error(f"Failed to connect to OpenSearch: {e}")
        raise
    return client


def list_unique_values(index_name: str, field_name: str, size: int = 100):
    """
    List unique values of a field using terms aggregation.

    :param index_name: OpenSearch index
    :param field_name: Field/column name
    :param size: Max number of unique values to fetch (default 100)
    :return: List of unique values
    """
    client = connect_opensearch()

    # Use .keyword subfield for text fields to enable aggregation
    aggregation_field = f"{field_name}.keyword" if not field_name.endswith('.keyword') else field_name

    query = {
        "size": 0,  # we don't need actual docs, just aggregation
        "aggs": {"unique_values": {"terms": {"field": aggregation_field, "size": size}}},  # max number of buckets
    }

    try:
        response = client.search(index=index_name, body=query)
        buckets = response["aggregations"]["unique_values"]["buckets"]
        return [bucket["key"] for bucket in buckets]
    except Exception as e:
        # If .keyword field doesn't exist, try the original field name
        if ".keyword" in aggregation_field:
            logger.warning(f"Field {aggregation_field} not found, trying {field_name}")
            query["aggs"]["unique_values"]["terms"]["field"] = field_name
            response = client.search(index=index_name, body=query)
            buckets = response["aggregations"]["unique_values"]["buckets"]
            return [bucket["key"] for bucket in buckets]
        else:
            raise e


if __name__ == "__main__":
    index = "vortex_lambda_execution_log_prod_2025_9,vortex_lambda_execution_log_prod_2025_8,vortex_lambda_execution_log_prod_2025_7"
    field = "lambda_name"  # for text fields, use the `.keyword` subfield
    unique_values = list_unique_values(index, field, size=1000)

    print(f"Unique values in '{field}':")
    for val in unique_values:
        print(val)
