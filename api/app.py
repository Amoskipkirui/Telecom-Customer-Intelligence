

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, render_template

from api.retention_db import (
    initialise_database,
    create_action,
    get_actions,
    update_action
)
app = Flask(__name__)

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_telecom_churn_model.pkl"
DATA_PATH = BASE_DIR / "data" / "customer_analytics_clean.csv"

# Load the trained model and customer data
model = joblib.load(MODEL_PATH)
customers = pd.read_csv(DATA_PATH)

# Standardise IDs for reliable customer searches
customers["customer_id"] = customers["customer_id"].astype(str)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "customers_loaded": len(customers)
    })


@app.route("/customer/<customer_id>", methods=["GET"])
def get_customer(customer_id):
    result = customers[
        customers["customer_id"] == str(customer_id)
    ]

    if result.empty:
        return jsonify({"error": "Customer not found"}), 404

    customer = result.iloc[0].to_dict()

    # Convert missing or non-JSON-compatible values safely
    customer = {
        key: None if pd.isna(value) else value
        for key, value in customer.items()
    }

    return jsonify(customer)


@app.route("/customers", methods=["GET"])
def list_customers():
    sample = customers[["customer_id"]].head(10)
    return jsonify(sample.to_dict(orient="records"))



@app.route("/predict/<customer_id>", methods=["GET"])
def predict_churn(customer_id):
    result = customers[
        customers["customer_id"] == str(customer_id)
    ]

    if result.empty:
        return jsonify({"error": "Customer not found"}), 404

    # Get customer features in the same format used during training.
    customer = result.iloc[[0]].copy()
    X_customer = customer.drop(
        columns=["customer_id", "churn"],
        errors="ignore"
    )

    # Generate prediction and probability from the saved pipeline.
    prediction = str(model.predict(X_customer)[0])
    probabilities = model.predict_proba(X_customer)[0]
    classes = list(model.classes_)

    churn_index = classes.index("Yes")
    churn_probability = float(probabilities[churn_index])

    # Business thresholds for the prototype.
    if churn_probability >= 0.60:
        risk_level = "High"
    elif churn_probability >= 0.30:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Example retention actions based on customer information.
    customer_row = result.iloc[0]

    if customer_row["contract_type"] == "Month-to-month":
        action = (
            "Offer a suitable longer-term plan with clear pricing."
        )
    elif customer_row["auto_pay"] == "No":
        action = (
            "Offer help setting up automatic payments, if suitable."
        )
    elif customer_row["complaints_count"] > 0:
        action = (
            "Review recent complaints and follow up with the customer."
        )
    else:
        action = (
            "Contact the customer to understand their needs "
            "and discuss relevant loyalty benefits."
        )

    return jsonify({
        "customer_id": str(customer_id),
        "predicted_churn": prediction,
        "churn_probability": round(churn_probability, 4),
        "churn_probability_percent": round(
            churn_probability * 100, 2
        ),
        "risk_level": risk_level,
        "recommended_action": action
    })

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/summary", methods=["GET"])
def business_summary():
    total_customers = len(customers)

    churn_labels = customers["churn"].astype(str).str.strip().str.lower()
    churned_customers = int((churn_labels == "yes").sum())

    churn_rate = (
        churned_customers / total_customers * 100
        if total_customers else 0
    )

    average_monthly_charges = float(
        customers["monthly_charges_qar"].mean()
    )

    return jsonify({
        "total_customers": total_customers,
        "churned_customers": churned_customers,
        "actual_churn_rate_percent": round(churn_rate, 2),
        "average_monthly_charges_qar": round(
            average_monthly_charges, 2
        )
    })


REVENUE_PATH = (
    BASE_DIR / "data" / "monthly_customer_revenue_clean.csv"
)

revenue = pd.read_csv(REVENUE_PATH)
revenue["customer_id"] = revenue["customer_id"].astype(str)
revenue["month"] = pd.to_datetime(revenue["month"], errors="coerce")

revenue["monthly_revenue_qar"] = pd.to_numeric(
    revenue["monthly_revenue_qar"],
    errors="coerce"
)

@app.route("/risk-summary", methods=["GET"])
def risk_summary():
    # Prepare the same input features used by the trained model.
    X = customers.drop(
        columns=["customer_id", "churn"],
        errors="ignore"
    )

    # Calculate churn probabilities for all customers.
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    churn_index = classes.index("Yes")

    churn_probabilities = probabilities[:, churn_index]

    # Apply the existing business risk thresholds.
    low_risk = int((churn_probabilities < 0.30).sum())

    medium_risk = int(
        (
            (churn_probabilities >= 0.30)
            & (churn_probabilities < 0.60)
        ).sum()
    )

    high_risk = int((churn_probabilities >= 0.60).sum())

    total = len(customers)

    return jsonify({
        "total_customers": total,
        "low_risk": low_risk,
        "medium_risk": medium_risk,
        "high_risk": high_risk,
        "low_risk_percent": round(low_risk / total * 100, 2) if total else 0,
        "medium_risk_percent": round(medium_risk / total * 100, 2) if total else 0,
        "high_risk_percent": round(high_risk / total * 100, 2) if total else 0
    })



@app.route("/retention-priority", methods=["GET"])
def retention_priority():
    selected_risk = request.args.get("risk", "All").title()

    if selected_risk not in ["All", "High", "Medium", "Low"]:
        return jsonify({
            "error": "Invalid risk level. Use All, High, Medium, or Low."
        }), 400

    # 1. Prepare model input features.
    X = customers.drop(
        columns=["customer_id", "churn"],
        errors="ignore"
    )

    # 2. Predict churn probability for every customer.
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    churn_index = classes.index("Yes")

    result = customers[["customer_id"]].copy()
    result["churn_probability"] = (
        probabilities[:, churn_index] * 100
    )

    # 3. Assign churn risk using the existing thresholds.
    result["risk_level"] = result["churn_probability"].apply(
        lambda p: "High" if p >= 60
        else "Medium" if p >= 30
        else "Low"
    )

    # 4. Get each customer's latest available monthly revenue.
    latest_revenue = (
        revenue.sort_values("month")
        .drop_duplicates("customer_id", keep="last")
        [["customer_id", "monthly_revenue_qar", "month"]]
        .rename(columns={
            "monthly_revenue_qar": "monthly_revenue_qar",
            "month": "revenue_month"
        })
    )

    # 5. Attach revenue to each customer.
    result = result.merge(
        latest_revenue,
        on="customer_id",
        how="left"
    )

    # Customers without revenue records should not be assumed
    # to generate zero revenue; mark their revenue as unavailable.
    result["monthly_revenue_qar"] = (
        result["monthly_revenue_qar"].round(2)
    )

    # 6. Calculate a simple monthly revenue exposure estimate.
    # This is a prioritisation aid, not confirmed lost revenue.
    result["estimated_monthly_revenue_exposure_qar"] = (
        result["monthly_revenue_qar"]
        * result["churn_probability"] / 100
    ).round(2)

    # 7. Add recommended retention actions.
    customer_actions = customers[
        ["customer_id", "contract_type", "auto_pay", "complaints_count"]
    ].copy()

    customer_actions["recommended_action"] = customer_actions.apply(
        lambda row:
            "Contact customer urgently and review retention options"
            if row["contract_type"] == "Month-to-month"
            and row["complaints_count"] > 0
            else "Offer a suitable longer-term plan"
            if row["contract_type"] == "Month-to-month"
            else "Offer automatic payment assistance"
            if row["auto_pay"] == "No"
            else "Review complaints and follow up"
            if row["complaints_count"] > 0
            else "Discuss relevant loyalty benefits",
        axis=1
    )

    result = result.merge(
        customer_actions[["customer_id", "recommended_action"]],
        on="customer_id",
        how="left"
    )

    # 8. Apply the selected risk filter.
    if selected_risk != "All":
        result = result[
            result["risk_level"] == selected_risk
        ]

    # 9. Rank customers by estimated monthly revenue exposure.
    # Customers without revenue records appear after those with data.
    result = result.sort_values(
        by=[
            "estimated_monthly_revenue_exposure_qar",
            "churn_probability"
        ],
        ascending=[False, False],
        na_position="last"
    ).head(10).copy()

    result["retention_priority"] = [
        f"Priority {i}" for i in range(1, len(result) + 1)
    ]

    result["churn_probability"] = (
        result["churn_probability"].round(2)
    )

    result["revenue_month"] = (
        result["revenue_month"].dt.strftime("%Y-%m")
    )

    # 10. Return the results to the dashboard.
    return jsonify(result.to_dict(orient="records"))



@app.route("/revenue-at-risk", methods=["GET"])
def revenue_at_risk():
    # Get churn probabilities for all customers.
    X = customers.drop(
        columns=["customer_id", "churn"],
        errors="ignore"
    )

    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    churn_index = classes.index("Yes")

    risk_data = pd.DataFrame({
        "customer_id": customers["customer_id"].astype(str),
        "churn_probability": probabilities[:, churn_index] * 100
    })

    # Use the existing high-risk threshold of 60%.
    high_risk_customers = risk_data[
        risk_data["churn_probability"] >= 60
    ]

    # Get the latest available revenue record for each customer.
    latest_revenue = (
        revenue.dropna(
            subset=["customer_id", "month", "monthly_revenue_qar"]
        )
        .sort_values("month")
        .drop_duplicates("customer_id", keep="last")
    )

    # Match high-risk customers to their latest revenue.
    matched = high_risk_customers.merge(
        latest_revenue[
            ["customer_id", "month", "monthly_revenue_qar"]
        ],
        on="customer_id",
        how="left"
    )

    matched["monthly_revenue_qar"] = matched[
        "monthly_revenue_qar"
    ].fillna(0)

    total_customers_at_risk = len(high_risk_customers)
    matched_customers = int(
        matched["monthly_revenue_qar"].gt(0).sum()
    )

    estimated_revenue_at_risk = float(
        matched["monthly_revenue_qar"].sum()
    )

    latest_month = (
        latest_revenue["month"].max()
        if not latest_revenue.empty
        else None
    )

    return jsonify({
        "high_risk_customers": total_customers_at_risk,
        "customers_with_revenue": matched_customers,
        "estimated_monthly_revenue_at_risk_qar": round(
            estimated_revenue_at_risk, 2
        ),
        "revenue_data_month": (
            latest_month.strftime("%Y-%m")
            if pd.notna(latest_month)
            else None
        ),
        "definition": (
            "Sum of latest available monthly revenue for customers "
            "whose model-estimated churn probability is at least 60%. "
            "This is potential exposure, not confirmed lost revenue."
        )
    })

@app.route("/revenue-risk-breakdown", methods=["GET"])
def revenue_risk_breakdown():
    # Estimate churn probability for every customer.
    X = customers.drop(
        columns=["customer_id", "churn"],
        errors="ignore"
    )

    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    churn_index = classes.index("Yes")

    risk_data = pd.DataFrame({
        "customer_id": customers["customer_id"].astype(str),
        "churn_probability": probabilities[:, churn_index] * 100
    })

    # Assign the existing business risk categories.
    risk_data["risk_level"] = risk_data["churn_probability"].apply(
        lambda p: "High" if p >= 60
        else "Medium" if p >= 30
        else "Low"
    )

    # Keep the latest available revenue record for each customer.
    latest_revenue = (
        revenue.dropna(
            subset=[
                "customer_id",
                "month",
                "monthly_revenue_qar"
            ]
        )
        .sort_values("month")
        .drop_duplicates("customer_id", keep="last")
    )

    # Match customer risk to revenue.
    combined = risk_data.merge(
        latest_revenue[
            ["customer_id", "month", "monthly_revenue_qar"]
        ],
        on="customer_id",
        how="left"
    )

    results = []

    for risk_level in ["High", "Medium", "Low"]:
        group = combined[combined["risk_level"] == risk_level]

        revenue_total = group["monthly_revenue_qar"].sum(min_count=1)

        results.append({
            "risk_level": risk_level,
            "customer_count": int(len(group)),
            "customers_with_revenue": int(
                group["monthly_revenue_qar"].notna().sum()
            ),
            "estimated_monthly_revenue_qar": round(
                float(revenue_total)
                if pd.notna(revenue_total) else 0,
                2
            )
        })

    return jsonify({
        "revenue_data_month": (
            latest_revenue["month"].max().strftime("%Y-%m")
            if not latest_revenue.empty else None
        ),
        "breakdown": results,
        "definition": (
            "Revenue is the latest available monthly revenue per customer, "
            "grouped by model-estimated churn risk. These figures indicate "
            "revenue exposure, not confirmed losses."
        )
    })

# Create a retention action
@app.route("/retention-actions", methods=["POST"])
def add_retention_action():
    data = request.get_json(silent=True) or {}

    customer_id = str(data.get("customer_id", "")).strip()
    action_type = str(data.get("action_type", "")).strip()
    notes = str(data.get("notes", "")).strip()

    if not customer_id or not action_type:
        return jsonify({
            "error": "customer_id and action_type are required."
        }), 400

    # Confirm the customer exists in the existing dataset.
    if customer_id not in set(customers["customer_id"]):
        return jsonify({
            "error": "Customer not found."
        }), 404

    allowed_actions = {
        "Phone Call",
        "Complaint Follow-up",
        "Retention Offer",
        "Account Review"
    }

    if action_type not in allowed_actions:
        return jsonify({
            "error": "Invalid action_type."
        }), 400

    if len(notes) > 1000:
        return jsonify({
            "error": "Notes must be 1000 characters or fewer."
        }), 400

    action_id = create_action(customer_id, action_type, notes)

    return jsonify({
        "message": "Retention action created successfully.",
        "action_id": action_id,
        "customer_id": customer_id,
        "action_type": action_type,
        "status": "Pending"
    }), 201


# Retrieve retention actions, optionally for one customer
@app.route("/retention-actions", methods=["GET"])
def list_retention_actions():
    customer_id = request.args.get("customer_id", "").strip()

    if customer_id:
        if customer_id not in set(customers["customer_id"]):
            return jsonify({"error": "Customer not found."}), 404

    return jsonify(get_actions(customer_id or None))


# Update the status of an existing retention action
@app.route("/retention-actions/<int:action_id>", methods=["PATCH"])
def edit_retention_action(action_id):
    data = request.get_json(silent=True) or {}

    status = str(data.get("status", "")).strip()
    notes = data.get("notes")

    if not status:
        return jsonify({"error": "status is required."}), 400

    if notes is not None:
        if not isinstance(notes, str) or len(notes) > 1000:
            return jsonify({
                "error": "Notes must be text of 1000 characters or fewer."
            }), 400

    try:
        updated = update_action(action_id, status, notes)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    if not updated:
        return jsonify({"error": "Retention action not found."}), 404

    return jsonify({
        "message": "Retention action updated successfully.",
        "action_id": action_id,
        "status": status
    })
initialise_database()

if __name__ == "__main__":
    app.run(debug=True)
