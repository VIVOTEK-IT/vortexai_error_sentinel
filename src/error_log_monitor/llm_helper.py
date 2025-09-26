"""
LLM Helper Module

This module provides a flexible and extensible system for managing OpenAI function calls.
It uses a registry-based approach that makes it easy to add new functions.

Key Features:
1. Function Registry: Central registry for all available functions
2. Dynamic Function Definitions: Automatically generates OpenAI function definitions
3. Handler Pattern: Each function has its own handler for clean separation
4. Decorator Support: Easy registration using @function_tool decorator
5. Runtime Management: Add/remove functions at runtime

Adding New Functions:

Method 1 - Using Decorator (Recommended):
```python
@function_tool(
    name="my_new_function",
    description="Does something useful",
    parameters={
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "First parameter"}
        },
        "required": ["param1"]
    }
)
def my_new_function_handler(function_args: dict, rds_client) -> str:
    param1 = function_args.get("param1")
    # Your implementation here
    return json.dumps({"result": "success"})
```

Method 2 - Using register_function():
```python
def my_handler(function_args: dict, rds_client) -> str:
    # Implementation here
    pass

register_function(
    name="my_function",
    description="Does something",
    parameters={...},
    handler_func_name="my_handler"
)
```

Method 3 - Direct Registry Modification:
```python
FUNCTION_REGISTRY["my_function"] = {
    "description": "Does something",
    "parameters": {...},
    "handler": "my_handler"
}
```

All handler functions must follow this signature:
def handler_name(function_args: dict, rds_client) -> str:
    # Must return JSON string
    return json.dumps(result)
"""

import json
import logging
import os
import datetime
import openai
from error_log_monitor.opensearch_client import OpenSearchClient
from error_log_monitor.rds_client import create_rds_client
from error_log_monitor.config import SystemConfig

logger = logging.getLogger(__name__)

PRICE_PER_1M_INPUT_TOKENS = 0.05  # USD
PRICE_PER_1M_OUTPUT_TOKENS = 0.40  # USD
THRESHOLD_USD = 2.0
INDEX_NAME = "openai_usage"


# Function registry for easy addition of new functions
FUNCTION_REGISTRY = {
    "get_camera_info": {
        "description": "get a row of camera_info by thing name.",
        "parameters": {
            "type": "object",
            "properties": {
                "mac": {
                    "type": "string",
                    "description": "MAC address with or without channel. e.g.0002D1B20120_ch24 or 0002D1B20120",
                },
            },
            "required": ["mac"],
        },
        "handler": "get_camera_info_handler",
    },
    "check_object_trace_partition_exists": {
        "description": "check if a object_trace's partition table exists.",
        "parameters": {
            "type": "object",
            "properties": {
                "partition_id": {
                    "type": "string",
                    "description": (
                        "partition id is composed by {partition_group_id}_{year}_{month}_{day}, "
                        "e.g. 1_2025_1_1. It is used to compose a partition table name of object_trace, "
                        'e.g. "OBJECT_TRACE_1_2025_1_1".'
                    ),
                },
            },
            "required": ["partition_id"],
        },
        "handler": "check_object_trace_partition_exists_handler",
    },
    "get_partition_info": {
        "description": "get a row of partition_info by mac and partition_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "mac": {
                    "type": "string",
                    "description": "MAC address with or without channel. e.g.0002D1B20120_ch24 or 0002D1B20120",
                },
                "partition_id": {
                    "type": "string",
                    "description": (
                        "partition id is composed by {partition_group_id}_{year}_{month}_{day}, "
                        "e.g. 1_2025_1_1. It is used to compose a partition table name of object_trace, "
                        'e.g. "OBJECT_TRACE_1_2025_1_1".'
                    ),
                },
            },
            "required": ["mac", "partition_id"],
        },
        "handler": "get_partition_info_handler",
    },
    "get_object_trace": {
        "description": "get a row of object_trace by mac and partition_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "mac": {
                    "type": "string",
                    "description": "MAC address with or without channel. e.g.0002D1B20120_ch24 or 0002D1B20120",
                },
                "partition_id": {
                    "type": "string",
                    "description": (
                        "partition id is composed by {partition_group_id}_{year}_{month}_{day}, "
                        "e.g. 1_2025_1_1. It is used to compose a partition table name of object_trace, "
                        'e.g. "OBJECT_TRACE_1_2025_1_1".'
                    ),
                },
            },
            "required": ["mac", "partition_id"],
        },
        "handler": "get_object_trace_handler",
    },
}


def get_function_definitions():
    """Generate function definitions from the registry."""
    return [
        {
            "type": "function",
            "name": name,
            "description": func_info["description"],
            "parameters": func_info["parameters"],
        }
        for name, func_info in FUNCTION_REGISTRY.items()
    ]


# Get the function definitions for OpenAI API
FUNCTION_CALL_DEFINITIONS_PROMPT = get_function_definitions()


def register_function(name: str, description: str, parameters: dict, handler_func_name: str):
    """
    Register a new function in the registry.

    Args:
        name: Function name
        description: Function description
        parameters: OpenAI function parameters schema
        handler_func_name: Name of the handler function
    """
    FUNCTION_REGISTRY[name] = {"description": description, "parameters": parameters, "handler": handler_func_name}
    # Update the function definitions
    global FUNCTION_CALL_DEFINITIONS_PROMPT
    FUNCTION_CALL_DEFINITIONS_PROMPT = get_function_definitions()


def unregister_function(name: str):
    """Remove a function from the registry."""
    if name in FUNCTION_REGISTRY:
        del FUNCTION_REGISTRY[name]
        # Update the function definitions
        global FUNCTION_CALL_DEFINITIONS_PROMPT
        FUNCTION_CALL_DEFINITIONS_PROMPT = get_function_definitions()


def list_registered_functions():
    """List all registered function names."""
    return list(FUNCTION_REGISTRY.keys())


def function_tool(name: str, description: str, parameters: dict):
    """
    Decorator to register a function as a tool.

    Usage:
        @function_tool(
            name="my_function",
            description="Does something useful",
            parameters={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"}
                },
                "required": ["param1"]
            }
        )
        def my_function_handler(function_args: dict, rds_client) -> str:
            # Implementation here
            pass
    """

    def decorator(handler_func):
        register_function(name, description, parameters, handler_func.__name__)
        return handler_func

    return decorator


# Example of how to add a new function using the decorator
@function_tool(
    name="get_system_status",
    description="Get the current system status and health metrics",
    parameters={
        "type": "object",
        "properties": {
            "component": {
                "type": "string",
                "description": "Specific component to check (optional, checks all if not provided)",
            }
        },
        "required": [],
    },
)
def get_system_status_handler(function_args: dict, rds_client) -> str:
    """Handler for get_system_status function."""
    component = function_args.get("component")

    # Example implementation
    status = {
        "overall_status": "healthy",
        "components": {"database": "connected", "opensearch": "connected", "vector_db": "connected"},
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
    }

    if component:
        status["requested_component"] = component
        status["component_status"] = status["components"].get(component, "unknown")

    return json.dumps(status)


# Individual function handlers
def get_camera_info_handler(function_args: dict, rds_client) -> str:
    """Handler for get_camera_info function."""
    mac = function_args.get("mac")
    if not mac:
        return json.dumps({"error": "mac parameter is required"})

    result = rds_client.get_camera_info(mac)
    return json.dumps(result)


def check_object_trace_partition_exists_handler(function_args: dict, rds_client) -> str:
    """Handler for check_object_trace_partition_exists function."""
    partition_id = function_args.get("partition_id")
    if not partition_id:
        return json.dumps({"error": "partition_id parameter is required"})

    # This would need to be implemented in RDS client
    return json.dumps({"status": "not_implemented", "message": "Partition check not implemented"})


def get_partition_info_handler(function_args: dict, rds_client) -> str:
    """Handler for get_partition_info function."""
    mac = function_args.get("mac")
    partition_id = function_args.get("partition_id")
    if not mac or not partition_id:
        return json.dumps({"error": "mac and partition_id parameters are required"})

    result = rds_client.get_partition_info(partition_id)
    return json.dumps(result)


def get_object_trace_handler(function_args: dict, rds_client) -> str:
    """Handler for get_object_trace function."""
    mac = function_args.get("mac")
    partition_id = function_args.get("partition_id")
    if not mac or not partition_id:
        return json.dumps({"error": "mac and partition_id parameters are required"})

    result = rds_client.get_object_trace(mac, partition_id)
    return json.dumps(result)


def execute_tool_call(tool_call, config: SystemConfig) -> str:
    """Execute a tool call using the function registry."""
    try:
        # Handle different tool call object structures
        if hasattr(tool_call, 'function'):
            # Old format: tool_call.function.name
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
        else:
            # New format: tool_call.name directly
            function_name = tool_call.name
            function_args = json.loads(tool_call.arguments)

        logger.info(f"Executing tool call: {function_name} with args: {function_args}")

        # Check if function exists in registry
        if function_name not in FUNCTION_REGISTRY:
            return json.dumps({"error": f"Unknown function: {function_name}"})

        # Get handler function name from registry
        handler_name = FUNCTION_REGISTRY[function_name]["handler"]

        # Get the handler function
        handler_func = globals().get(handler_name)
        if not handler_func:
            return json.dumps({"error": f"Handler function {handler_name} not found"})

        # Create RDS client for tool execution
        rds_client = create_rds_client(config.rds)

        # Execute the handler function
        result = handler_func(function_args, rds_client)
        return result

    except Exception as e:
        # Handle different tool call object structures for error logging
        try:
            if hasattr(tool_call, 'function'):
                function_name = tool_call.function.name
            else:
                function_name = tool_call.name
        except:
            function_name = "unknown"

        logger.error(f"Error executing tool call {function_name}: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


def calc_cost(usage):
    in_cost = usage.input_tokens * (PRICE_PER_1M_INPUT_TOKENS / 1_000_000)
    out_cost = usage.output_tokens * (PRICE_PER_1M_OUTPUT_TOKENS / 1_000_000)
    return in_cost + out_cost


def log_usage_to_opensearch(resp, opensearch_client):
    today = datetime.date.today().isoformat()
    doc = {
        "date": today,
        "model": resp.model,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "total_tokens": resp.usage.total_tokens,
        "cost_usd": calc_cost(resp.usage),
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc),
    }
    opensearch_client.index(index=INDEX_NAME, body=doc)


def check_today_usage(opensearch_client):
    today = datetime.date.today().isoformat()
    query = {"size": 0, "query": {"term": {"date": today}}, "aggs": {"total_cost": {"sum": {"field": "cost_usd"}}}}
    try:
        resp = opensearch_client.search(index=INDEX_NAME, body=query)
        return resp["aggregations"]["total_cost"]["value"]
    except Exception as e:
        logging.error(f"Error checking today's usage: {str(e)}")
        return 0.0


def call_llm(client, shared_messages, model_name, config: SystemConfig):
    opensearch_client = OpenSearchClient(config.opensearch).get_es_conn()
    today_cost = check_today_usage(opensearch_client)
    if today_cost >= THRESHOLD_USD:
        raise RuntimeError(f"⚠️ Exceeding openai daily quota ${today_cost:.2f}")
    response = client.responses.create(
        model=model_name,
        input=shared_messages,
        tools=FUNCTION_CALL_DEFINITIONS_PROMPT,
    )
    log_usage_to_opensearch(response, opensearch_client)
    shared_messages += response.output
    second_round = False
    for item in response.output:
        if item.type == "function_call":
            result = execute_tool_call(item, config)
            shared_messages.append({"type": "function_call_output", "call_id": item.call_id, "output": str(result)})
            second_round = True
    if second_round:
        print('===============================================')
        for item in response.output:
            if item.type == "message":
                print(item.content[0].text)
        print('===============================================')
        try:
            response2 = call_llm(client, shared_messages, model_name, config)
            return response2
        except Exception as e:
            print(e)
            raise e
    return response


def llm_chat(shared_messages, model_name, config: SystemConfig):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    result = call_llm(client, shared_messages, model_name, config)
    for item in result.output:
        if item.type == "message":
            return item
    return result


if __name__ == "__main__":
    shared_messages = [
        {
            "role": "system",
            "content": (
                "You are a poetic assistant, skilled in explaining complex programming concepts " "with creative flair."
            ),
        },
    ]
    shared_messages.append(
        {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
    )
    answer = llm_chat(shared_messages, "")
    shared_messages.append({"role": "assistant", "content": answer})
    # print("*" * 100)
    # print(shared_messages)
    # print(f"Q: {question}\nA: {answer}")
    # print("*" * 100)
    print(shared_messages, answer)
