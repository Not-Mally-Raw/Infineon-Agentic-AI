import os
import json
import re
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd
import requests
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


def _number_code_lines(code: str) -> str:
    lines = code.splitlines()
    return "\n".join(f"{idx + 1}: {line}" for idx, line in enumerate(lines))


def _extract_json_from_text(text: str) -> Optional[str]:
    if not text:
        return None

    tagged = re.search(r"<json>\s*(.*?)\s*</json>", text, re.DOTALL)
    if tagged:
        return tagged.group(1).strip()

    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if obj_match:
        return obj_match.group(0)

    arr_match = re.search(r"\[.*\]", text, re.DOTALL)
    if arr_match:
        return arr_match.group(0)

    return None


def _safe_json_loads(text: str, default: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return default


def _groq_generate(client: ChatGroq, prompt: str, max_new_tokens: int) -> str:
    response = client.invoke(
        [HumanMessage(content=prompt)],
        max_tokens=max_new_tokens,
    )
    return response.content


def _mcp_search(server_url: str, query: str) -> List[Dict[str, Any]]:
    try:
        response = requests.post(
            f"{server_url}/api/search",
            json={"query": query},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        print(f"MCP error: {response.status_code} - {response.text}")
        return []
    except Exception as exc:
        print(f"Error calling MCP: {exc}")
        return []


def mcp_test(server_url: str) -> bool:
    try:
        return len(_mcp_search(server_url, "test")) > 0
    except Exception:
        return False


State = TypedDict(
    "State",
    {
        "code_id": str,
        "code": str,
        "context": str,
        "extracted": Dict[str, Any],
        "suspicious": List[Dict[str, Any]],
        "mcp_docs": Dict[str, Any],
        "result": Dict[str, Any],
    },
)


def _build_extract_prompt(code: str) -> str:
    numbered_code = _number_code_lines(code)
    return f"""
Extract structured API usage from this RDI code.

Return ONLY valid JSON with this exact shape:
{{
  \"api_calls\": [\"rdi.smartVec()\", \"rdi.dc().vForce()\"],
  \"parameters\": {{\"vecEditMode\": \"TA::VECD\", \"vForce\": \"31 V\"}},
  \"functions\": [\"smartVec\", \"dc\", \"vForce\", \"execute\"],
  \"lifecycle_markers\": [\"RDI_BEGIN\", \"RDI_END\"],
  \"lifecycle_order\": [\"RDI_BEGIN\", \"RDI_END\"],
  \"line_mapping\": {{\"1\": \"rdi.smartVec().vecEditMode(TA::VECD)\", \"3\": \"RDI_BEGIN()\"}},
  \"pins\": [\"A\", \"dig1\"],
  \"variables\": [\"sLabel\", \"rt\"]
}}

Rules:
- Preserve exact parameter values.
- Use 1-based line numbers as shown in the code block.
- Include all function calls and lifecycle markers.
- Wrap the JSON in <json>...</json> with no extra text.

CODE:
```cpp
{numbered_code}
```
"""


def _build_detect_prompt(code: str, extracted: Dict[str, Any], context: str) -> str:
    numbered_code = _number_code_lines(code)
    return f"""
Detect likely RDI bugs using the context hint and extracted elements.

Return ONLY valid JSON array. Each item must be:
{{
  \"pattern\": \"short_identifier\",
  \"line\": 1,
  \"found_value\": \"exact value in code\",
  \"reason\": \"why this is likely wrong\",
  \"mcp_query\": \"5-10 word doc query\"
}}

Bug families:
- wrong parameter value/range
- wrong function name
- wrong parameter order
- lifecycle order errors (RDI_BEGIN/RDI_END)
- method chain misuse (.write vs .execute)
- pin/port mismatch
- wrong enum/mode

Only include high-confidence issues. Use 1-based line numbers.
Wrap the JSON in <json>...</json> with no extra text.

CONTEXT:
{context}

EXTRACTED:
{json.dumps(extracted, indent=2)}

CODE:
```cpp
{numbered_code}
```
"""


def _build_explain_prompt(
    code: str,
    primary_bug: Dict[str, Any],
    top_doc: Dict[str, Any],
    context: str,
) -> str:
    numbered_code = _number_code_lines(code)
    return f"""
Write a concise bug explanation grounded in the documentation.

Return ONLY plain text (2-3 sentences).

Bug:
- pattern: {primary_bug['pattern']}
- line: {primary_bug['line']}
- found_value: {primary_bug['found_value']}
- reason: {primary_bug['reason']}

Doc snippet:
{top_doc.get('text', 'No documentation found')}

Context:
{context}

Code:
```cpp
{numbered_code}
```
"""


def _node_extract(state: State, client: ChatGroq) -> State:
    prompt = _build_extract_prompt(state["code"])
    text = _groq_generate(client, prompt, max_new_tokens=900)
    json_text = _extract_json_from_text(text)
    extracted = _safe_json_loads(json_text, {}) if json_text else {}
    state["extracted"] = extracted if isinstance(extracted, dict) else {}
    return state


def _node_detect(state: State, client: ChatGroq) -> State:
    prompt = _build_detect_prompt(state["code"], state.get("extracted", {}), state["context"])
    text = _groq_generate(client, prompt, max_new_tokens=1200)
    json_text = _extract_json_from_text(text)
    suspicious = _safe_json_loads(json_text, []) if json_text else []
    state["suspicious"] = suspicious if isinstance(suspicious, list) else []
    return state


def _node_retrieve(state: State, server_url: str) -> State:
    documentation: Dict[str, Any] = {}
    for pattern in state.get("suspicious", []):
        query = pattern.get("mcp_query", "")
        if not query:
            continue
        print(f"  📚 Querying MCP: '{query}'")
        mcp_results = _mcp_search(server_url, query)
        if mcp_results:
            print(f"     ✓ Got {len(mcp_results)} results (top score: {mcp_results[0]['score']:.3f})")
        else:
            print("     ✗ No results from MCP")
        documentation[pattern.get("pattern", "unknown")] = {
            "query": query,
            "mcp_results": mcp_results,
            "top_doc": mcp_results[0] if mcp_results else None,
            "top_3_docs": mcp_results[:3] if mcp_results else [],
        }
    state["mcp_docs"] = documentation
    return state


def _node_explain(state: State, client: ChatGroq) -> State:
    suspicious = state.get("suspicious", [])
    if not suspicious:
        state["result"] = {
            "line": -1,
            "explanation": "No bugs detected",
            "confidence": 0.0,
        }
        return state

    primary_bug = suspicious[0]
    bug_docs = state.get("mcp_docs", {}).get(primary_bug.get("pattern", ""), {}) or {}
    top_doc = bug_docs.get("top_doc") or {}

    prompt = _build_explain_prompt(state["code"], primary_bug, top_doc, state["context"])
    explanation = _groq_generate(client, prompt, max_new_tokens=400).strip()

    state["result"] = {
        "line": primary_bug.get("line", -1),
        "explanation": explanation,
        "confidence": top_doc.get("score", 0.5) if top_doc else 0.5,
    }
    return state


def build_graph(api_key: str, model_main: str, model_detector: str, server_url: str):
    client = ChatGroq(
        api_key=api_key,
        model=model_main,
        temperature=0.2,
    )

    graph = StateGraph(State)

    graph.add_node("extract", lambda state: _node_extract(state, client))
    graph.add_node("detect", lambda state: _node_detect(state, client))
    graph.add_node("retrieve", lambda state: _node_retrieve(state, server_url))
    graph.add_node("explain", lambda state: _node_explain(state, client))

    graph.set_entry_point("extract")
    graph.add_edge("extract", "detect")
    graph.add_edge("detect", "retrieve")
    graph.add_edge("retrieve", "explain")
    graph.add_edge("explain", END)

    return graph.compile()


def process_code(
    app,
    code_id: str,
    code: str,
    context: str,
) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"Processing ID: {code_id}")
    print(f"{'=' * 60}")

    print("🔍 [1/4] Extracting code elements...")
    print("🐛 [2/4] Detecting suspicious patterns...")
    print("📚 [3/4] Querying MCP for documentation...")
    print("✍️  [4/4] Generating explanation...")

    final_state = app.invoke(
        {
            "code_id": code_id,
            "code": code,
            "context": context,
            "extracted": {},
            "suspicious": [],
            "mcp_docs": {},
            "result": {},
        }
    )

    result = final_state.get("result", {})
    print(f"\n✓ Bug found on line {result.get('line', -1)}")
    print(f"  Confidence: {result.get('confidence', 0.0):.2f}")
    print(f"  Explanation: {str(result.get('explanation', ''))[:100]}...")

    return {
        "ID": code_id,
        "Bug Line": result.get("line", -1),
        "Explanation": result.get("explanation", "No bugs detected"),
    }


def main():
    api_key = os.getenv("GROQ_API_KEY", "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")
    model_main = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    model_detector = os.getenv("GROQ_MODEL_DETECTOR", model_main)
    server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8003")
    dataset_path = "samples.csv"
    output_path = "output.csv"

    if not api_key:
        print("ERROR: Set GROQ_API_KEY environment variable")
        return

    print("\n" + "=" * 60)
    print("RDI BUG HUNTER - Using Existing MCP Server")
    print("=" * 60 + "\n")

    if mcp_test(server_url):
        print(f"✓ MCP server connected at {server_url}")
    else:
        print(f"✗ WARNING: Cannot connect to MCP server at {server_url}")

    app = build_graph(api_key, model_main, model_detector, server_url)

    print(f"\n📂 Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"   Loaded {len(df)} code samples")

    results: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        result = process_code(
            app,
            code_id=str(row["ID"]),
            code=row["Code"],
            context=row["Context"],
        )
        results.append(result)

    print(f"\n💾 Saving results to {output_path}...")
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)

    print(f"\n✓ Done! Processed {len(results)} samples")
    print(f"  Output saved to: {output_path}")

    print("\n📊 Sample Results:")
    print(output_df.head())


if __name__ == "__main__":
    main()
