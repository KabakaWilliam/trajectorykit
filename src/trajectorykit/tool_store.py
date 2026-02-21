import requests
from typing import Any, Callable, Dict, List, Optional
from .utils import SUPPORTED_LANGUAGES, API_TIMEOUT, MAX_RETRIES, INITIAL_RETRY_DELAY
from .config import MAX_RECURSION_DEPTH, SUB_AGENT_TURN_BUDGET, CONTEXT_WINDOW
import logging
import os
import threading
import time
import traceback
import uuid
import json
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, just continue
    pass

logger = logging.getLogger(__name__)


# Define tool implementations
def get_current_time() -> str:
    """Get the current date and time"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add_numbers(a: float, b: float) -> str:
    """Add two numbers together"""
    result = a + b
    return f"{a} + {b} = {result}"

def search_web(q: str, num_results: int = 5) -> str:
    """
    Execute a Google search via SerpAPI and return structured results.
    
    Args:
        q: Search query string
        num_results: Number of results to return (default: 5)
    
    Returns:
        Formatted string with top search results or error message
    """
    api_key = os.getenv("SERP_API_KEY", "")
    
    if not api_key:
        return "Error: SerpAPI key not configured. Set SERP_API_KEY environment variable."
    
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": q,
        "api_key": api_key,
        "num": min(num_results, 10),  # SerpAPI default is 10, cap at 10
        "hl": "en",
        "gl": "us",
    }
    
    try:
        logger.info(f"Executing search query: {q}")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if "error" in data:
            return f"Search API Error: {data.get('error', 'Unknown error')}"
        
        # Extract organic results
        results = data.get("organic_results", [])
        
        if not results:
            return f"No results found for query: {q}"
        
        # Format results for readability
        formatted_results = f"Search Results for '{q}':\n\n"
        for i, result in enumerate(results[:num_results], 1):
            title = result.get("title", "No title")
            link = result.get("link", "No link")
            snippet = result.get("snippet", "No snippet")
            formatted_results += f"{i}. {title}\n   URL: {link}\n   {snippet}\n\n"
        
        logger.info(f"Successfully retrieved {len(results[:num_results])} search results")
        return formatted_results
        
    except requests.exceptions.Timeout:
        error_msg = f"Search timeout: Query '{q}' took too long to complete"
        logger.error(error_msg)
        return error_msg
    except requests.exceptions.ConnectionError:
        error_msg = "Search error: Could not connect to SerpAPI. Check internet connection."
        logger.error(error_msg)
        return error_msg
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return "Search error: Invalid API key. Check SERPAPI_KEY."
        elif e.response.status_code == 403:
            return "Search error: API key not authorized for this request."
        elif e.response.status_code == 429:
            return "Search error: Rate limit exceeded. Please try again later."
        else:
            error_msg = f"Search HTTP error {e.response.status_code}"
            logger.error(error_msg)
            return error_msg
    except json.JSONDecodeError:
        error_msg = "Search error: Invalid JSON response from API"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Search error: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        return error_msg

# tool_store.py — add alongside execute_code

def spawn_agent(
    task: str,
    context: Optional[str] = None,
    turn_length: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    _depth: int = 0,
) -> str:
    """
    Spawn a sub-agent to handle a subtask. Calls dispatch() recursively.
    Returns the sub-agent's final response as a string.
    """
    from .agent import dispatch   # lazy import to avoid circular dependency
    from .config import MAX_RECURSION_DEPTH, get_model_profile, MODEL_NAME

    resolved_model = model or MODEL_NAME
    profile = get_model_profile(resolved_model)

    if turn_length is None:
        turn_length = SUB_AGENT_TURN_BUDGET
    if max_tokens is None:
        max_tokens = profile["context_window"]

    if _depth >= MAX_RECURSION_DEPTH:
        return json.dumps({
            "response": f"[RECURSION LIMIT] Depth {_depth} reached (max {MAX_RECURSION_DEPTH}). Cannot spawn further sub-agents.",
            "turns_used": 0,
            "tool_calls_made": 0,
        }), None

    full_input = f"{context}\n\nTASK: {task}" if context else task

    if _depth > 0:
        logger.info(f"Spawning sub-agent at depth {_depth}: {task[:80]}...")

    result = dispatch(
        user_input=full_input,
        turn_length=turn_length,
        verbose=False,           # sub-agents run silently
        max_tokens=profile["context_window"],
        temperature=temperature,
        model=model,
        reasoning_effort=reasoning_effort,
        _depth=_depth + 1,       # increment depth for recursive calls
    )

    # Extract child trace from the result
    child_trace = result.get("trace", None)

    output = json.dumps({
        "response": result["final_response"],
        "turns_used": result["turns"],
        "tool_calls_made": result["tool_calls"],
        "depth": _depth + 1,
    })

    return output, child_trace

def execute_code(
    completion: str,
    stdin: Optional[str] = '',
    compile_timeout: int = 10,
    run_timeout: int = 5,
    memory_limit_mb: int = 128,
    language: str = "python",
    files: Optional[dict[str, str]] = None,      # filename -> base64 content
    fetch_files: Optional[list[str]] = None,      # list of filenames to return
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    
    from .config import SANDBOX_FUSION_URL

    code = completion
    if "```python" in completion:
        code = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        # Handle cases like ```\ncode\n```
        parts = completion.split("```")
        if len(parts) >= 2:
            code = parts[1]
            # Remove potential language specifier like 'python\n'
            if "\n" in code:
                first_line, rest = code.split("\n", 1)
                if first_line.strip().isalpha():  # Simple check for language name
                    code = rest
    else:
        return 0.0, [{"error": "Invalid completion (missing code block)"}]
    
    request_id = str(uuid.uuid4())  # <-- Generate request_id internally
    log_prefix = f"[Request ID: {request_id}] "  # <-- Create log prefix

    if language not in SUPPORTED_LANGUAGES:
        error_msg = f"{log_prefix}Unsupported language: {language}"
        logger.error(error_msg)
        return None, error_msg

    payload = json.dumps(
        {
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
            "code": code,
            "stdin": stdin,
            "memory_limit_MB": memory_limit_mb,
            "language": language,  # Use the passed language parameter
            "files": files or {},
            "fetch_files": fetch_files or [],
        }
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    # Calculate a reasonable request timeout based on compile/run timeouts plus a buffer
    request_timeout = compile_timeout + run_timeout + API_TIMEOUT

    last_error = None  # Store the last error encountered

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling sandbox API at {SANDBOX_FUSION_URL}"
            )  # <-- Use internal log_prefix
            response = requests.post(
                SANDBOX_FUSION_URL,
                headers=headers,
                data=payload,
                timeout=request_timeout,  # Use the calculated timeout
            )

            # Check for Gateway Timeout (504) specifically for retrying
            if response.status_code == 504:
                last_error = (
                    f"{log_prefix}API Request Error: Gateway Timeout (504) on attempt "
                    f"{attempt + 1}/{MAX_RETRIES}"
                )  # <-- Use internal log_prefix
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:  # Don't sleep after the last attempt
                    # Calculate increasing delay (e.g., 1s, 2s, 4s, ...) or (1s, 2s, 3s, ...)
                    # Simple linear increase: delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    # Exponential backoff: delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)  # Using linear increase for simplicity
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")  # <-- Use internal log_prefix
                    time.sleep(delay)
                continue  # Go to the next retry attempt

            # Check for other HTTP errors (e.g., 4xx, other 5xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(
                f"{log_prefix}Sandbox API call successful on attempt {attempt + 1}"
            )  # <-- Use internal log_prefix
            return response.json(), None

        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on non-504 request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}Sandbox API call failed. Last error: {last_error}")  # <-- Use internal log_prefix
    # Return the error message without the prefix, as the caller doesn't need the internal ID
    # Ensure API call failure returns error message, leading to -1 in check_correctness
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"

def execute_code_wrapper(**kwargs):
    """Wrapper to call execute_code and format output for vLLM.
    Returns (output_str, None) — no child trace for code execution."""
    try:
        # execute_code returns a tuple: (result_dict, error_string)
        sandbox_result, error = execute_code(**kwargs)
        
        # If there was an error
        if error:
            return f"ERROR: {error}", None
        
        # If no result
        if sandbox_result is None:
            return "ERROR: No result returned from sandbox", None
        
        # Parse the sandbox result
        if isinstance(sandbox_result, dict):
            status = sandbox_result.get("status", "Unknown")
            
            # Check if execution succeeded
            if status == "Success":
                run_result = sandbox_result.get("run_result", {})
                return_code = run_result.get("return_code", -1)
                stdout = run_result.get("stdout", "")
                stderr = run_result.get("stderr", "")
                exec_time = run_result.get("execution_time", 0)
                
                # Format output for vLLM
                output = f"Exit Code: {return_code}\n"
                output += f"Execution Time: {exec_time:.3f}s\n"
                
                if stdout:
                    output += f"STDOUT:\n{stdout}"
                if stderr:
                    output += f"STDERR:\n{stderr}"
                
                # Include any fetched files (base64-encoded) from the sandbox
                fetched = sandbox_result.get("files", {})
                if fetched:
                    output += "\nFETCHED FILES:\n"
                    for fname, b64data in fetched.items():
                        output += f"--- {fname} (base64) ---\n{b64data}\n"
                
                # Warn if files were requested but none came back
                requested_files = kwargs.get("fetch_files", [])
                if requested_files and not fetched:
                    output += (
                        f"\nWARNING: You requested fetch_files={requested_files} but no files were returned. "
                        "Make sure your code writes these files to the working directory "
                        "(e.g., use plt.savefig('output.png') instead of plt.show())."
                    )
                
                result = output if output.strip() else "Code executed successfully with no output"
                return result, None
            else:
                # Execution failed — gather as much detail as possible
                message = sandbox_result.get("message", "")
                run_result = sandbox_result.get("run_result", {}) or {}
                run_status = run_result.get("status", "")
                stderr = run_result.get("stderr", "")
                stdout = run_result.get("stdout", "")
                return_code = run_result.get("return_code")
                exec_time = run_result.get("execution_time", 0)
                
                # Detect specific sandbox failure modes from run_result.status
                if run_status == "TimeLimitExceeded":
                    run_timeout = kwargs.get("run_timeout", 5)
                    error_parts = [
                        f"ERROR: Time limit exceeded — your code ran for {exec_time:.1f}s and was killed (limit: {run_timeout}s).",
                        "Your code took too long to execute. Common causes:",
                        "- Infinite loops (while True) or very long loops",
                        "- Blocking calls (time.sleep, input()) or network waits",
                        "- Computationally expensive operations",
                        "FIX: Rewrite your code to complete within the time limit. Do NOT use infinite loops in the sandbox.",
                    ]
                    if stdout:
                        error_parts.append(f"Partial STDOUT before kill:\n{stdout}")
                elif run_status == "MemoryLimitExceeded":
                    memory_limit = kwargs.get("memory_limit_mb", 128)
                    error_parts = [
                        f"ERROR: Memory limit exceeded (limit: {memory_limit}MB).",
                        "Your code used too much memory. Reduce data size or optimize memory usage.",
                    ]
                else:
                    error_parts = [f"ERROR: Execution failed (status: {status})"]
                    if run_status:
                        error_parts.append(f"Run status: {run_status}")
                    if message:
                        error_parts.append(f"Message: {message}")
                    if return_code is not None:
                        error_parts.append(f"Return code: {return_code}")
                    if stderr:
                        error_parts.append(f"STDERR:\n{stderr}")
                    if stdout:
                        error_parts.append(f"STDOUT:\n{stdout}")
                    if not message and not stderr and not run_status:
                        error_parts.append("No error details provided by sandbox. Check your code for syntax errors or missing imports.")
                
                return "\n".join(error_parts), None
        
        # Fallback
        return f"Unexpected result format: {str(sandbox_result)[:200]}", None
        
    except Exception as e:
        return f"ERROR: {str(e)}", None

def get_current_time_wrapper(**kwargs):
    """Wrapper for get_current_time tool. Returns (output, None)."""
    try:
        result = get_current_time()
        return f"Current time: {result}", None
    except Exception as e:
        return f"ERROR: {str(e)}", None

def add_numbers_wrapper(**kwargs):
    """Wrapper for add_numbers tool. Returns (output, None)."""
    try:
        a = kwargs.get("a")
        b = kwargs.get("b")
        
        if a is None or b is None:
            return "ERROR: Both 'a' and 'b' parameters are required", None
        
        result = add_numbers(a, b)
        return result, None
    except Exception as e:
        return f"ERROR: {str(e)}", None

def search_web_wrapper(**kwargs):
    """Wrapper for search_web tool. Returns (output, None)."""
    try:
        q = kwargs.get("q")
        num_results = kwargs.get("num_results", 5)

        return search_web(q=q, num_results=num_results), None
    except Exception as e:
        return f"ERROR: {str(e)}", None

def spawn_agent_wrapper(_depth: int = 0, _model: Optional[str] = None, _reasoning_effort: Optional[str] = None, **kwargs):
    """Wrapper for spawn_agent tool. Injects _depth, _model, _reasoning_effort from the parent dispatch loop.
    Returns (output_str, child_trace) where child_trace is an EpisodeTrace."""
    try:
        task = kwargs.get("task")
        if not task:
            return "ERROR: 'task' parameter is required", None

        from .config import SUB_AGENT_TURN_BUDGET
        output, child_trace = spawn_agent(
            task=task,
            context=kwargs.get("context"),
            turn_length=kwargs.get("turn_length", SUB_AGENT_TURN_BUDGET),
            max_tokens=kwargs.get("max_tokens"),
            temperature=kwargs.get("temperature"),
            model=_model,
            reasoning_effort=_reasoning_effort,
            _depth=_depth,
        )
        return output, child_trace
    except Exception as e:
        return f"ERROR: {str(e)}", None

def search_available_tools_wrapper(**kwargs):
    """Wrapper for search_available_tools. Returns (output, None)."""
    try:
        tool_name = kwargs.get("tool_name")
        if tool_name:
            # Return full schema for the requested tool
            for tool in TOOLS:
                if tool["function"]["name"] == tool_name:
                    return json.dumps(tool, indent=2), None
            return f"ERROR: No tool named '{tool_name}'. Call search_available_tools with no arguments to see all available tools.", None
        else:
            # Return compact summary: name + description for each tool
            summary = []
            for tool in TOOLS:
                func = tool["function"]
                name = func["name"]
                desc = func["description"].split('\n')[0]  # first line only
                required = func.get("parameters", {}).get("required", [])
                summary.append(f"- {name} (required args: {required})\n  {desc}")
            return "Available tools:\n\n" + "\n\n".join(summary), None
    except Exception as e:
        return f"ERROR: {str(e)}", None


def recall_wrapper(scratchpad: dict, **kwargs):
    """Retrieve data from the scratchpad. Returns (output, None)."""
    try:
        key = kwargs.get("key")
        query = kwargs.get("query")

        if not scratchpad:
            return "Scratchpad is empty. No data has been stored yet.", None

        if key:
            # Exact key lookup
            if key in scratchpad:
                return scratchpad[key], None
            available = ", ".join(scratchpad.keys())
            return f"No entry for '{key}'. Available keys: {available}", None

        if query:
            # Substring search across all stored values
            query_lower = query.lower()
            matches = []
            for k, v in scratchpad.items():
                idx = v.lower().find(query_lower)
                if idx != -1:
                    start = max(0, idx - 50)
                    end = min(len(v), idx + len(query) + 100)
                    snippet = v[start:end]
                    matches.append(f"  {k}: ...{snippet}...")
            if matches:
                return f"Found '{query}' in {len(matches)} entries:\n" + "\n".join(matches), None
            available = ", ".join(scratchpad.keys())
            return f"No entries contain '{query}'. Available keys: {available}", None

        # No args — list all keys with previews
        lines = []
        for k, v in scratchpad.items():
            lines.append(f"  {k} ({len(v)} chars): {v[:80]}...")
        return "Stored data:\n" + "\n".join(lines), None
    except Exception as e:
        return f"ERROR: {str(e)}", None


def dispatch_tool_call(tool_name: str, tool_args: dict, _depth: int = 0, model: Optional[str] = None, reasoning_effort: Optional[str] = None, scratchpad: Optional[dict] = None):
    """Route tool calls to appropriate wrapper function.
    
    All wrappers return (output_str, child_trace_or_None).
    
    Args:
        tool_name: Name of the tool to call
        tool_args: Arguments the model provided for the tool
        _depth: Current recursion depth (injected by the agent loop, invisible to the model)
        model: Model name to propagate to sub-agents
        reasoning_effort: Reasoning effort level to propagate to sub-agents
        scratchpad: Shared scratchpad dict for storing/retrieving large outputs
    
    Returns:
        Tuple of (output_string, child_trace) where child_trace is an
        EpisodeTrace if the tool was spawn_agent, otherwise None.
    """
    if tool_name == "execute_code":
        return execute_code_wrapper(**tool_args)
    elif tool_name == "get_current_time":
        return get_current_time_wrapper(**tool_args)
    elif tool_name == "add_numbers":
        return add_numbers_wrapper(**tool_args)
    elif tool_name == "search_web":
        return search_web_wrapper(**tool_args)
    elif tool_name == "spawn_agent":
        return spawn_agent_wrapper(_depth=_depth, _model=model, _reasoning_effort=reasoning_effort, **tool_args)
    elif tool_name == "search_available_tools":
        return search_available_tools_wrapper(**tool_args)
    elif tool_name == "recall":
        return recall_wrapper(scratchpad=scratchpad or {}, **tool_args)
    else:
        return f"ERROR: Unknown tool '{tool_name}'", None


# Define the tool schemas for vLLM
TOOLS = [
{
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": (
            "Execute code in a sandboxed environment with support for multiple languages, "
            "input/output handling, file I/O, and resource limits. "
            "The code should be provided in a markdown code block (e.g., ```python code here ```). "
            "Returns execution results including stdout, stderr, exit status, and any requested output files.\n\n"
            "FILE HANDLING:\n"
            "- `files`: Upload input files (e.g. CSVs, images, data) into the sandbox before execution. "
            "Provide as a dict of {filename: base64_encoded_string}. "
            "Files are placed in the working directory and can be read by name in your code.\n"
            "- `fetch_files`: Retrieve output files generated by your code after execution (e.g. plots, reports, ZIPs). "
            "Provide a list of filenames your code will write. "
            "They are returned as base64-encoded strings in the response under the 'files' key.\n\n"
            "EXAMPLE WORKFLOW: To generate a chart → write ```python ... plt.savefig('plot.png') ``` "
            "and set fetch_files=['plot.png']. The plot will be returned as base64 for display."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "completion": {
                    "type": "string",
                    "description": (
                        "Code to execute. MUST be wrapped in markdown code blocks "
                        "(```python ... ``` or ``` ... ```). "
                        "The code will be automatically extracted from the code block."
                    )
                },
                "stdin": {
                    "type": "string",
                    "description": "Standard input to pipe into the program (default: '').",
                    "default": ""
                },
                "language": {
                    "type": "string",
                    "description": "Programming language to use (default: 'python').",
                    "default": "python",
                    "enum": SUPPORTED_LANGUAGES
                },
                "files": {
                    "type": "object",
                    "description": (
                        "Input files to upload into the sandbox before execution. "
                        "Keys are filenames (e.g. 'data.csv'), values are base64-encoded file contents. "
                        "Files are written to the working directory and accessible by name in your code."
                    ),
                    "additionalProperties": {
                        "type": "string",
                        "description": "Base64-encoded file content."
                    },
                    "default": {}
                },
                "fetch_files": {
                    "type": "array",
                    "description": (
                        "List of filenames to retrieve from the sandbox after execution. "
                        "Your code must write these files to the working directory. "
                        "They are returned as base64-encoded strings in the response under the 'files' key. "
                        "Use this to retrieve generated plots, CSVs, ZIPs, PDFs, etc."
                    ),
                    "items": {
                        "type": "string",
                        "description": "Filename to fetch (e.g. 'output.png', 'result.csv')."
                    },
                    "default": []
                },
                "compile_timeout": {
                    "type": "integer",
                    "description": "Compilation timeout in seconds (default: 10).",
                    "default": 10
                },
                "run_timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (default: 5).",
                    "default": 5
                },
                "memory_limit_mb": {
                    "type": "integer",
                    "description": "Memory limit in megabytes (default: 128).",
                    "default": 128
                }
            },
            "required": ["completion"]
        }
    }
},
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time in YYYY-MM-DD HH:MM:SS format",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the internet using Google via SerpAPI. Returns formatted search results with titles, URLs, and snippets. "
                "Use this to find current information, answer factual questions, or research topics. "
                "Automatically handles errors like rate limits, invalid keys, and timeouts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {
                        "type": "string",
                        "description": "The search query string"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10, default: 5)",
                        "default": 5
                    }
                },
                "required": ["q"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers together and return the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number to add"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number to add"
                    }
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_agent",
            "description": (
                "DELEGATE a subtask to an independent sub-agent. This is your PRIMARY tool for complex tasks. "
                "The sub-agent gets its own fresh context, tools, and turn budget. It does the work and returns ONLY its final result — "
                "keeping your context clean and small.\n\n"
                "YOU MUST USE THIS TOOL WHEN:\n"
                "- The user asks to compare, research, or analyze 2+ items → spawn one sub-agent PER item\n"
                "- A subtask needs search + code execution → delegate it, don't do it yourself\n"
                "- You're about to make 3+ tool calls for one part of the task → that's a sub-agent\n\n"
                "HOW TO WRITE THE TASK STRING:\n"
                "The sub-agent has NO context from your conversation. The task string is ALL it gets.\n"
                "- BAD: 'Research card A' (too vague)\n"
                "- GOOD: 'Find the ATK, DEF, Level, Type, and Attribute of the Yu-Gi-Oh card Blue-Eyes White Dragon. "
                "Search the web for accurate stats. Return the results as a JSON object with keys: name, atk, def, level, type, attribute.'\n\n"
                "PATTERN FOR MULTI-ITEM TASKS:\n"
                "1. spawn_agent(task='[detailed task for item 1, specify return format]')\n"
                "2. spawn_agent(task='[detailed task for item 2, specify return format]')\n"
                "3. Collect results, then use execute_code yourself to visualize/synthesize"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Clear description of the subtask the sub-agent should accomplish"
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Optional background context or data the sub-agent needs. "
                            "Include any relevant information from the current conversation."
                        )
                    },
                    "turn_length": {
                        "type": "integer",
                        "description": f"Maximum turns for the sub-agent (default: {SUB_AGENT_TURN_BUDGET})",
                        "default": SUB_AGENT_TURN_BUDGET
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": f"Maximum tokens per generation for the sub-agent (default: {CONTEXT_WINDOW})",
                        "default": CONTEXT_WINDOW
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature for the sub-agent (default: 0.7)",
                        "default": 0.7
                    }
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_available_tools",
            "description": (
                "Look up the tools you have available and their parameter schemas. "
                "Call with no arguments to get a compact summary of all tools. "
                "Call with a tool_name to get the full JSON schema for that specific tool, "
                "including all parameters, types, defaults, and descriptions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": (
                            "Name of a specific tool to get its full schema. "
                            "Omit to list all available tools with short descriptions."
                        )
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Retrieve data from the scratchpad. Large tool outputs are automatically "
                "stored here to keep your context clean. "
                "Call with no args to list all stored keys with previews. "
                "Call with key to retrieve a specific entry. "
                "Call with query to search across all stored values by substring."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Exact key name to retrieve (shown in [Stored as 'key'] receipts)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Substring to search for across all stored values"
                    }
                },
                "required": []
            }
        }
    }
]
