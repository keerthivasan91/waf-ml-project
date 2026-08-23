from fastapi import FastAPI

app = FastAPI(
    title="Protected Demo Application",
    description="Demo backend protected by the Hybrid Intelligent WAF",
    version="1.0"
)

# ============================================================
# HEALTH SIMULATION
# ============================================================

_health_state = {
    "forced_error_rate": None
}


@app.post("/simulate/breach")
def simulate_breach(error_rate: float = 0.15):
    """
    Force /health to report a breach-level error rate.
    Used to demonstrate the WAF health-feedback mechanism.
    """
    _health_state["forced_error_rate"] = error_rate

    return {
        "status": "breach_simulated",
        "error_rate": error_rate
    }


@app.post("/simulate/recover")
def simulate_recover():
    """Clear the simulated health breach."""
    _health_state["forced_error_rate"] = None

    return {
        "status": "recovered"
    }


# ============================================================
# BASIC APPLICATION
# ============================================================

@app.get("/")
def home():
    return {
        "application": "Protected Online Store",
        "status": "running",
        "message": "Backend is protected by the Hybrid Intelligent WAF"
    }


@app.get("/health")
def health():

    if _health_state["forced_error_rate"] is not None:
        return {
            "status": "degraded",
            "error_rate": _health_state["forced_error_rate"]
        }

    return {
        "status": "healthy",
        "error_rate": 0.0
    }


@app.get("/hello")
def hello():
    return {
        "message": "Hello from the protected application"
    }


# ============================================================
# PRODUCT API
# ============================================================

@app.get("/api/products")
def products(
    category: str = "",
    page: int = 1,
    limit: int = 10
):
    return {
        "endpoint": "products",
        "category": category,
        "page": page,
        "limit": limit,
        "products": [
            {
                "id": 101,
                "name": "Laptop",
                "category": "electronics"
            },
            {
                "id": 102,
                "name": "Wireless Mouse",
                "category": "electronics"
            }
        ]
    }


@app.get("/api/products/search")
def search_products(q: str = ""):
    return {
        "endpoint": "product_search",
        "query": q,
        "results": [
            {
                "id": 101,
                "name": "Laptop"
            }
        ]
    }


@app.get("/api/products/{product_id}")
def product_details(product_id: int):
    return {
        "endpoint": "product_details",
        "product_id": product_id,
        "name": "Laptop",
        "price": 59999
    }


# ============================================================
# USER API
# ============================================================

@app.get("/api/users/profile")
def profile(user_id: int = 1):
    return {
        "endpoint": "user_profile",
        "user_id": user_id,
        "name": "Demo User",
        "role": "customer"
    }


@app.post("/api/users/login")
def login(
    username: str = "",
    password: str = ""
):
    return {
        "endpoint": "login",
        "username": username,
        "authenticated": True
    }


# ============================================================
# ORDER API
# ============================================================

@app.get("/api/orders")
def orders(
    user_id: int = 1,
    status: str = "all"
):
    return {
        "endpoint": "orders",
        "user_id": user_id,
        "status": status,
        "orders": [
            {
                "order_id": 5001,
                "status": "delivered"
            }
        ]
    }


@app.get("/api/orders/details")
def order_details(order_id: int = 5001):
    return {
        "endpoint": "order_details",
        "order_id": order_id,
        "status": "delivered"
    }


# ============================================================
# CART API
# ============================================================

@app.get("/api/cart")
def cart(user_id: int = 1):
    return {
        "endpoint": "cart",
        "user_id": user_id,
        "items": [
            {
                "product_id": 101,
                "quantity": 1
            }
        ]
    }


# ============================================================
# REVIEW / COMMENT API
# ============================================================

@app.get("/api/reviews")
def reviews(
    product_id: int = 101,
    sort: str = "recent"
):
    return {
        "endpoint": "reviews",
        "product_id": product_id,
        "sort": sort,
        "reviews": [
            {
                "rating": 5,
                "comment": "Good product"
            }
        ]
    }


# ============================================================
# ADMIN API
# ============================================================

@app.get("/api/admin/dashboard")
def admin_dashboard():
    return {
        "endpoint": "admin_dashboard",
        "users": 1250,
        "orders": 4821,
        "status": "operational"
    }


@app.get("/api/admin/users")
def admin_users(
    search: str = "",
    page: int = 1
):
    return {
        "endpoint": "admin_users",
        "search": search,
        "page": page
    }


# ============================================================
# FILE / RESOURCE ENDPOINT
# Useful for demonstrating LFI / path traversal protection
# ============================================================

@app.get("/api/files/view")
def view_file(path: str = ""):
    return {
        "endpoint": "file_view",
        "requested_file": path
    }


@app.get("/api/files/download")
def download_file(file: str = ""):
    return {
        "endpoint": "file_download",
        "requested_file": file
    }


# ============================================================
# COMMAND / SYSTEM TEST ENDPOINT
# Useful only for controlled WAF demonstration
# ============================================================

@app.get("/api/system/check")
def system_check(value: str = ""):
    return {
        "endpoint": "system_check",
        "value": value
    }


@app.get("/api/system/run")
def system_run(command: str = ""):
    return {
        "endpoint": "system_run",
        "command": command
    }


# ============================================================
# CONTACT API
# ============================================================

@app.get("/api/contact")
def contact(
    subject: str = "",
    message: str = ""
):
    return {
        "endpoint": "contact",
        "subject": subject,
        "message": message,
        "status": "received"
    }