"""Mock ecommerce tools exposed through OpenAI function calling."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


def _json_response(payload: dict[str, Any]) -> str:
    """Return compact UTF-8 JSON for a tool result message."""
    return json.dumps(payload, ensure_ascii=False)


_ORDER_FIXTURES: dict[str, dict[str, Any]] = {
    "ORD-1001": {
        "status": "已发货",
        "product_name": "蓝牙耳机 Pro",
        "paid_amount": 199.0,
        "order_time": "2026-08-14 09:20:00",
    },
    "ORD-1002": {
        "status": "待支付",
        "product_name": "机械键盘",
        "paid_amount": 0.0,
        "order_time": "2026-08-15 16:45:00",
    },
    "EC-1001": {
        "status": "已发货",
        "product_name": "蓝牙耳机",
        "paid_amount": 129.0,
        "order_time": "2026-08-15 10:30:00",
    },
}

_LOGISTICS_FIXTURES: dict[str, str] = {
    "SF1234567890": "已到达杭州分拨中心，预计明日送达",
    "YT9876543210": "已从上海仓发出，运输中",
}

PRODUCTS_PATH = Path(__file__).with_name("products.json")
PRODUCT_SEARCH_TERMS = (
    "商品",
    "规格",
    "参数",
    "尺码",
    "尺寸",
    "鞋码",
    "脚长",
    "胸围",
    "材质",
    "材料",
    "兼容",
    "防水",
    "清洗",
    "保养",
    "保修",
    "售后",
    "退货",
    "退款",
    "换货",
    "政策",
    "faq",
)


def query_order_status(order_id: str) -> str:
    """Return mock order status data as a standard JSON string."""
    if not isinstance(order_id, str) or not order_id.strip():
        return _json_response({"success": False, "error": "order_id 不能为空"})

    normalized_order_id = order_id.strip()
    order = _ORDER_FIXTURES.get(normalized_order_id)
    if order is None:
        return _json_response(
            {
                "success": False,
                "order_id": normalized_order_id,
                "error": "未找到该订单，请核对订单号",
            }
        )

    return _json_response(
        {"success": True, "order_id": normalized_order_id, **order}
    )


def query_logistics(tracking_no: str) -> str:
    """Return a mock latest logistics node as a standard JSON string."""
    if not isinstance(tracking_no, str) or not tracking_no.strip():
        return _json_response(
            {"success": False, "error": "tracking_no 不能为空"}
        )

    normalized_tracking_no = tracking_no.strip()
    latest_node = _LOGISTICS_FIXTURES.get(normalized_tracking_no)
    if latest_node is None:
        return _json_response(
            {
                "success": False,
                "tracking_no": normalized_tracking_no,
                "error": "未找到该物流单号，请核对物流单号",
            }
        )

    return _json_response(
        {
            "success": True,
            "tracking_no": normalized_tracking_no,
            "latest_node": latest_node,
        }
    )


def create_human_ticket(user_id: str, issue_type: str, summary: str) -> str:
    """Create a simulated high-priority ticket for human support escalation."""
    fields = {
        "user_id": user_id,
        "issue_type": issue_type,
        "summary": summary,
    }
    if any(not isinstance(value, str) or not value.strip() for value in fields.values()):
        return _json_response(
            {
                "success": False,
                "error": "user_id、issue_type 和 summary 均不能为空",
            }
        )

    ticket_id = f"HT-{datetime.now().strftime('%Y%m%d')}-{uuid4().hex[:8].upper()}"
    return _json_response(
        {
            "success": True,
            "ticket_id": ticket_id,
            "status": "已创建",
            "routing": "人工客服",
            "priority": "high",
            "user_id": user_id.strip(),
            "issue_type": issue_type.strip(),
            "summary": summary.strip(),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "message": "工单已成功派给人工客服，请保留工单号以便后续查询。",
        }
    )


def _load_products() -> list[dict[str, Any]]:
    with PRODUCTS_PATH.open("r", encoding="utf-8") as file:
        products = json.load(file)
    if not isinstance(products, list):
        raise ValueError("products.json 必须包含商品对象数组")
    return products


def _search_terms(keyword: str) -> list[str]:
    normalized = keyword.strip().lower()
    terms = {normalized}
    terms.update(term for term in PRODUCT_SEARCH_TERMS if term in normalized)
    terms.update(part for part in normalized.split() if part)
    return sorted(terms, key=len, reverse=True)


def _product_search_text(product: dict[str, Any]) -> str:
    return json.dumps(product, ensure_ascii=False).lower()


def _faq_matches(product: dict[str, Any], terms: list[str]) -> list[dict[str, Any]]:
    matched_faq: list[dict[str, Any]] = []
    for faq in product.get("faq", []):
        faq_text = json.dumps(faq, ensure_ascii=False).lower()
        if any(term in faq_text for term in terms):
            matched_faq.append(faq)
    return matched_faq


def search_product_knowledge(keyword: str) -> str:
    """Search products.json and return matching specs, sizing, FAQs, and policy."""
    if not isinstance(keyword, str) or not keyword.strip():
        return _json_response(
            {"success": False, "error": "keyword 不能为空", "matches": []}
        )

    normalized_keyword = keyword.strip().lower()
    terms = _search_terms(keyword)
    matches: list[dict[str, Any]] = []

    for product in _load_products():
        search_text = _product_search_text(product)
        product_name = str(product.get("product_name", "")).lower()
        sku = str(product.get("sku", "")).lower()
        aliases = [str(alias).lower() for alias in product.get("aliases", [])]
        matched_terms = [term for term in terms if term and term in search_text]
        faq_matches = _faq_matches(product, terms)

        if not matched_terms and not faq_matches:
            continue

        score = len(matched_terms)
        if normalized_keyword in product_name or normalized_keyword in aliases:
            score += 8
        if normalized_keyword == sku:
            score += 10
        if faq_matches:
            score += 2

        matches.append(
            {
                "product_name": product.get("product_name"),
                "sku": product.get("sku"),
                "score": score,
                "matched_terms": matched_terms,
                "specifications": product.get("specifications", {}),
                "size_guide": product.get("size_guide", {}),
                "warranty": product.get("warranty"),
                "after_sales_policy": product.get("after_sales_policy", {}),
                "faq": faq_matches or product.get("faq", []),
            }
        )

    matches.sort(key=lambda item: item["score"], reverse=True)
    if not matches:
        return _json_response(
            {
                "success": False,
                "keyword": keyword,
                "error": "未找到匹配的商品知识，请补充商品名称、SKU 或具体问题",
                "matches": [],
            }
        )

    for match in matches:
        match.pop("score", None)

    return _json_response(
        {
            "success": True,
            "keyword": keyword,
            "matches": matches,
        }
    )


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_order_status",
            "description": "查询订单状态、商品名称、实付金额和下单时间。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "客户提供的订单号，例如 ORD-1001。",
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_logistics",
            "description": "查询物流单号对应的最新物流节点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_no": {
                        "type": "string",
                        "description": "客户提供的物流单号，例如 SF1234567890。",
                    }
                },
                "required": ["tracking_no"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_product_knowledge",
            "description": "检索商品规格、尺码建议、材质、保修、售后政策和常见问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "商品名称、SKU 或商品问题关键词，例如复古慢跑鞋尺码。",
                    }
                },
                "required": ["keyword"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_human_ticket",
            "description": "为敏感纠纷、投诉或超出知识库处理范围的问题创建人工客服工单。",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "客户用户 ID；CLI 没有用户身份信息时使用 anonymous。",
                    },
                    "issue_type": {
                        "type": "string",
                        "description": "问题类型，例如投诉诈骗、法律威胁、强烈要求人工客服或退款争议。",
                    },
                    "summary": {
                        "type": "string",
                        "description": "对客户问题的简洁事实摘要，不要编造订单或政策信息。",
                    },
                },
                "required": ["user_id", "issue_type", "summary"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "query_order_status": query_order_status,
    "query_logistics": query_logistics,
    "search_product_knowledge": search_product_knowledge,
    "create_human_ticket": create_human_ticket,
}


__all__ = [
    "TOOL_DEFINITIONS",
    "TOOL_HANDLERS",
    "query_logistics",
    "query_order_status",
    "search_product_knowledge",
    "create_human_ticket",
]
