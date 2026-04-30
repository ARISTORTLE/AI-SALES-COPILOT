from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_CANONICAL_COLUMNS = {"date", "product", "quantity", "revenue"}

ALIASES = {
    "date": "date",
    "order_date": "date",
    "invoice_date": "date",
    "day": "date",
    "order_id": "order_id",
    "orderid": "order_id",
    "product": "product",
    "item": "product",
    "sku": "product",
    "product_name": "product",
    "sub_category": "product",
    "sub-category": "product",
    "quantity": "quantity",
    "qty": "quantity",
    "units": "quantity",
    "sales_qty": "quantity",
    "revenue": "revenue",
    "sales": "revenue",
    "sales_amount": "revenue",
    "amount": "revenue",
    "gmv": "revenue",
    "profit": "profit",
    "margin": "profit",
    "gross_profit": "profit",
    "cost": "cost",
    "cogs": "cost",
    "unit_cost": "cost",
    "stock": "stock",
    "inventory": "stock",
    "on_hand": "stock",
    "stock_on_hand": "stock",
    "category": "category",
    "payment_mode": "payment_mode",
    "paymentmode": "payment_mode",
    "customer_name": "customer_name",
    "customername": "customer_name",
    "state": "state",
    "city": "city",
    "year_month": "year_month",
}


@dataclass
class SalesInsights:
    summary: dict[str, Any]
    weekday_performance: pd.DataFrame
    product_performance: pd.DataFrame
    daily_trend: pd.DataFrame
    restock_table: pd.DataFrame
    insight_cards: list[str]


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        clean = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        renamed[column] = ALIASES.get(clean, clean)
    return frame.rename(columns=renamed)


def prepare_sales_data(frame: pd.DataFrame) -> pd.DataFrame:
    data = _normalise_columns(frame).copy()

    missing = REQUIRED_CANONICAL_COLUMNS.difference(data.columns)
    if missing:
        required = ", ".join(sorted(REQUIRED_CANONICAL_COLUMNS))
        found = ", ".join(map(str, frame.columns))
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}. "
            f"Please upload data with columns like: {required}. Found: {found}"
        )

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "product"])

    for numeric_col in ["quantity", "revenue", "profit", "cost", "stock"]:
        if numeric_col in data.columns:
            data[numeric_col] = pd.to_numeric(data[numeric_col], errors="coerce")

    data["quantity"] = data["quantity"].fillna(0)
    data["revenue"] = data["revenue"].fillna(0)
    if "profit" not in data.columns:
        data["profit"] = np.nan
    if "cost" not in data.columns:
        data["cost"] = np.nan
    if "stock" not in data.columns:
        data["stock"] = np.nan

    # Prefer explicit profit when the dataset provides it. Fall back to revenue-cost.
    data["gross_margin"] = np.where(
        data["profit"].notna(),
        data["profit"],
        data["revenue"] - data["cost"].fillna(0),
    )
    data["margin_pct"] = np.where(
        data["revenue"] > 0,
        (data["gross_margin"] / data["revenue"]) * 100,
        0,
    )
    data["weekday"] = data["date"].dt.day_name()
    data["week"] = data["date"].dt.to_period("W").astype(str)
    return data.sort_values("date")


def _daily_trend(data: pd.DataFrame) -> pd.DataFrame:
    daily = (
        data.groupby("date", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            quantity=("quantity", "sum"),
            gross_margin=("gross_margin", "sum"),
        )
        .sort_values("date")
    )
    daily["day_index"] = np.arange(len(daily))

    if len(daily) >= 2:
        slope, intercept = np.polyfit(daily["day_index"], daily["revenue"], deg=1)
        forecast_index = len(daily) + 7
        predicted_revenue_7d = (slope * forecast_index) + intercept
    else:
        predicted_revenue_7d = daily["revenue"].iloc[-1] if not daily.empty else 0

    daily.attrs["predicted_revenue_7d"] = round(max(predicted_revenue_7d, 0), 2)
    return daily


def _weekday_performance(data: pd.DataFrame) -> pd.DataFrame:
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekday = (
        data.groupby("weekday", as_index=False)
        .agg(
            avg_revenue=("revenue", "mean"),
            avg_quantity=("quantity", "mean"),
            total_revenue=("revenue", "sum"),
            total_margin=("gross_margin", "sum"),
        )
    )
    weekday["weekday"] = pd.Categorical(weekday["weekday"], weekday_order, ordered=True)
    weekday = weekday.sort_values("weekday").reset_index(drop=True)
    return weekday


def _product_performance(data: pd.DataFrame) -> pd.DataFrame:
    product = (
        data.groupby("product", as_index=False)
        .agg(
            total_revenue=("revenue", "sum"),
            total_quantity=("quantity", "sum"),
            total_margin=("gross_margin", "sum"),
            avg_margin_pct=("margin_pct", "mean"),
            current_stock=("stock", "last"),
        )
        .sort_values(["total_margin", "total_revenue"], ascending=False)
        .reset_index(drop=True)
    )
    return product


def _restock_recommendations(data: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    recent_window_start = data["date"].max() - pd.Timedelta(days=14)
    recent = data[data["date"] >= recent_window_start]
    recent_velocity = (
        recent.groupby("product", as_index=False)
        .agg(last_14d_qty=("quantity", "sum"))
        .assign(avg_daily_demand=lambda frame: frame["last_14d_qty"] / 14)
    )

    restock = products.merge(recent_velocity, on="product", how="left")
    restock["avg_daily_demand"] = restock["avg_daily_demand"].fillna(0)
    restock["days_of_cover"] = np.where(
        restock["avg_daily_demand"] > 0,
        restock["current_stock"] / restock["avg_daily_demand"],
        np.nan,
    )
    restock["recommended_restock_qty"] = np.where(
        restock["avg_daily_demand"] > 0,
        np.ceil((restock["avg_daily_demand"] * 10) - restock["current_stock"].fillna(0)),
        0,
    )
    restock["recommended_restock_qty"] = restock["recommended_restock_qty"].clip(lower=0)

    def label_priority(row: pd.Series) -> str:
        if pd.isna(row["current_stock"]):
            return "Need stock column"
        if row["days_of_cover"] <= 3:
            return "Urgent"
        if row["days_of_cover"] <= 7:
            return "Watchlist"
        return "Healthy"

    restock["priority"] = restock.apply(label_priority, axis=1)
    return restock[
        [
            "product",
            "current_stock",
            "avg_daily_demand",
            "days_of_cover",
            "recommended_restock_qty",
            "priority",
        ]
    ].sort_values(["priority", "recommended_restock_qty"], ascending=[True, False])


def build_insights(data: pd.DataFrame) -> SalesInsights:
    daily = _daily_trend(data)
    weekday = _weekday_performance(data)
    products = _product_performance(data)
    restock = _restock_recommendations(data, products)

    total_revenue = float(data["revenue"].sum())
    total_margin = float(data["gross_margin"].sum())
    total_orders = int(len(data))
    predicted_revenue_7d = float(daily.attrs["predicted_revenue_7d"])

    insight_cards: list[str] = []
    if not weekday.empty:
        weakest_day = weekday.loc[weekday["avg_revenue"].idxmin()]
        strongest_day = weekday.loc[weekday["avg_revenue"].idxmax()]
        insight_cards.append(
            f"Sales are weakest on {weakest_day['weekday']} with average revenue of "
            f"Rs. {weakest_day['avg_revenue']:.0f}."
        )
        insight_cards.append(
            f"{strongest_day['weekday']} is your strongest day with average revenue of "
            f"Rs. {strongest_day['avg_revenue']:.0f}."
        )

    if not products.empty:
        top_margin = products.iloc[0]
        insight_cards.append(
            f"{top_margin['product']} delivers the highest total margin at "
            f"Rs. {top_margin['total_margin']:.0f}."
        )

    urgent = restock[restock["priority"] == "Urgent"]
    if not urgent.empty:
        focus_item = urgent.iloc[0]
        insight_cards.append(
            f"Restock {focus_item['product']} soon. It has only "
            f"{focus_item['days_of_cover']:.1f} days of cover left."
        )
    else:
        insight_cards.append("No product is in the urgent restock zone based on recent demand.")

    summary = {
        "total_revenue": round(total_revenue, 2),
        "total_margin": round(total_margin, 2),
        "total_orders": total_orders,
        "predicted_revenue_7d": round(predicted_revenue_7d, 2),
    }

    return SalesInsights(
        summary=summary,
        weekday_performance=weekday,
        product_performance=products,
        daily_trend=daily,
        restock_table=restock,
        insight_cards=insight_cards,
    )
