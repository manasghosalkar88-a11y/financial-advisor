import os
import io
import csv
import json
from math import pow

from flask import Flask, render_template, request, send_file
import numpy as np
import joblib
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# ---------- Flask setup ----------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# Load ML model (loan approval probability)
approval_model = joblib.load(os.path.join(BASE_DIR, "models", "approval_model.pkl"))


# ---------- Helper functions ----------

def calc_emi(loan_amount, interest_rate, tenure_years):
    """Standard EMI formula."""
    n = int(tenure_years * 12)
    if n <= 0:
        return 0.0

    r = interest_rate / (12 * 100)  # monthly rate

    if r == 0:
        return loan_amount / n

    emi = loan_amount * r * pow(1 + r, n) / (pow(1 + r, n) - 1)
    return emi


def create_amortization_schedule(loan_amount, interest_rate, tenure_years, emi):
    """Return normal amortization schedule (no prepayment)."""
    n = int(tenure_years * 12)
    r = interest_rate / (12 * 100)

    balance = loan_amount
    schedule = []
    total_interest = 0.0
    total_principal = 0.0

    for month in range(1, n + 1):
        interest_component = balance * r
        principal_component = emi - interest_component

        # Last payment adjustment
        if principal_component > balance:
            principal_component = balance
            emi_effective = principal_component + interest_component
        else:
            emi_effective = emi

        balance -= principal_component
        if balance < 0:
            balance = 0.0

        total_interest += interest_component
        total_principal += principal_component

        schedule.append({
            "month": month,
            "emi": round(emi_effective, 2),
            "interest": round(interest_component, 2),
            "principal": round(principal_component, 2),
            "balance": round(balance, 2)
        })

        if balance <= 0:
            break

    return schedule, total_interest, total_principal


def create_amortization_with_prepay(loan_amount, interest_rate, tenure_years, emi,
                                    prepay_amount, prepay_month):
    """Amortization schedule with one prepayment."""
    n = int(tenure_years * 12)
    r = interest_rate / (12 * 100)

    balance = loan_amount
    schedule = []
    total_interest = 0.0
    total_principal = 0.0

    for month in range(1, n + 1):
        if balance <= 0:
            break

        interest_component = balance * r
        principal_component = emi - interest_component

        if principal_component > balance:
            principal_component = balance
            emi_effective = principal_component + interest_component
        else:
            emi_effective = emi

        balance -= principal_component

        # Apply prepayment after EMI in selected month
        if month == prepay_month and prepay_amount > 0:
            if prepay_amount > balance:
                prepay_amount = balance
            balance -= prepay_amount
            total_principal += prepay_amount

        if balance < 0:
            balance = 0.0

        total_interest += interest_component
        total_principal += principal_component

        schedule.append({
            "month": month,
            "emi": round(emi_effective, 2),
            "interest": round(interest_component, 2),
            "principal": round(principal_component, 2),
            "balance": round(balance, 2)
        })

    return schedule, total_interest, total_principal


def get_credit_insights(score):
    """Map credit score to band + comment + tips."""
    score = float(score)

    if score >= 800:
        band = "Excellent"
        comment = "You are in an excellent range. Lenders usually offer the best rates."
        tips = [
            "Continue paying EMIs and bills on time.",
            "Avoid unnecessary new loans.",
            "Keep credit utilization under ~30%."
        ]
    elif score >= 750:
        band = "Very Good"
        comment = "Your score is very good. You qualify for most products at good rates."
        tips = [
            "Maintain on-time payments.",
            "Avoid closing old credit accounts suddenly.",
            "Limit number of unsecured loans."
        ]
    elif score >= 650:
        band = "Good"
        comment = "Your score is okay but not premium. Some offers may have higher interest."
        tips = [
            "Avoid missed EMIs for at least 6–12 months.",
            "Gradually reduce card balances.",
            "Avoid too many new applications."
        ]
    elif score >= 550:
        band = "Fair"
        comment = "Your score is in a risky zone. Lenders may charge higher interest."
        tips = [
            "Clear overdue EMIs and card dues first.",
            "Set up auto-debit to avoid misses.",
            "Pause new loan applications for a while."
        ]
    else:
        band = "Poor"
        comment = "Your score is low. Improve repayment history before big loans."
        tips = [
            "Pay all current dues on time for a long period.",
            "Close or settle old defaulted accounts responsibly.",
            "Avoid fresh credit until score improves."
        ]

    return band, comment, tips


# ---------- Routes: pages ----------

@app.route("/")
def home():
    return render_template("index.html", current_page="home")


@app.route("/about")
def about():
    return render_template("about.html", current_page="about")


@app.route("/compare", methods=["GET", "POST"])
def compare():
    """
    Compare two loan offers.
    """
    if request.method == "GET":
        return render_template("compare.html", current_page="compare", result=None)

    # POST: compare two offers
    try:
        loan1 = float(request.form["loan_amount1"])
        rate1 = float(request.form["interest_rate1"])
        tenure1 = float(request.form["tenure_years1"])
        fee1 = float(request.form.get("processing_fee1", 0) or 0)

        loan2 = float(request.form["loan_amount2"])
        rate2 = float(request.form["interest_rate2"])
        tenure2 = float(request.form["tenure_years2"])
        fee2 = float(request.form.get("processing_fee2", 0) or 0)

        emi1 = calc_emi(loan1, rate1, tenure1)
        sched1, tot_int1, _ = create_amortization_schedule(loan1, rate1, tenure1, emi1)
        total1 = tot_int1 + loan1 + fee1

        emi2 = calc_emi(loan2, rate2, tenure2)
        sched2, tot_int2, _ = create_amortization_schedule(loan2, rate2, tenure2, emi2)
        total2 = tot_int2 + loan2 + fee2

        if total1 < total2:
            better = "Offer 1 is cheaper overall."
            diff = total2 - total1
        elif total2 < total1:
            better = "Offer 2 is cheaper overall."
            diff = total1 - total2
        else:
            better = "Both offers have the same total cost."
            diff = 0.0

        result = {
            "emi1": round(emi1, 2),
            "total1": round(total1, 2),
            "emi2": round(emi2, 2),
            "total2": round(total2, 2),
            "better": better,
            "diff": round(diff, 2),
        }

        return render_template("compare.html", current_page="compare", result=result)

    except Exception as e:
        return f"<h3>Error in comparison: {e}</h3>"


# ---------- Routes: main prediction from home page ----------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        monthly_income = float(request.form["monthly_income"])
        age = float(request.form["age"])
        work_experience = float(request.form["work_experience"])
        existing_emi = float(request.form["existing_emi"])
        credit_score = float(request.form["credit_score"])
        loan_amount = float(request.form["loan_amount"])
        tenure_years = float(request.form["tenure_years"])
        interest_rate = float(request.form["interest_rate"])

        prepay_amount_str = request.form.get("prepay_amount", "").strip()
        prepay_month_str = request.form.get("prepay_month", "").strip()

        has_prepay = False
        prepay_amount = 0.0
        prepay_month = 0
        if prepay_amount_str and prepay_month_str:
            prepay_amount = float(prepay_amount_str)
            prepay_month = int(prepay_month_str)
            if prepay_amount > 0 and prepay_month > 0:
                has_prepay = True

        # EMI & schedule
        emi = calc_emi(loan_amount, interest_rate, tenure_years)
        schedule, total_interest, total_principal = create_amortization_schedule(
            loan_amount, interest_rate, tenure_years, emi
        )
        total_payment = total_interest + total_principal
        base_months = len(schedule)

        # Chart data
        balance_labels = [row["month"] for row in schedule]
        balance_data = [row["balance"] for row in schedule]
        balance_labels_json = json.dumps(balance_labels)
        balance_data_json = json.dumps(balance_data)

        # Simple risk score
        risk_score = (
            0.4 * (existing_emi / (monthly_income + 1)) * 100 +
            0.3 * (loan_amount / (monthly_income * 12 + 1)) * 10 -
            0.3 * ((credit_score - 550) / 300) * 100
        )

        # ML approval probability
        X_app = np.array([[
            monthly_income, existing_emi, loan_amount, tenure_years,
            credit_score, risk_score
        ]])
        approval_prob = approval_model.predict_proba(X_app)[0][1] * 100

        if risk_score < 30:
            risk_level = "Low"
        elif risk_score < 60:
            risk_level = "Medium"
        else:
            risk_level = "High"

        emi_to_income_pct = (emi / monthly_income) * 100 if monthly_income > 0 else 0

        if emi_to_income_pct < 30:
            affordability_msg = "Great! Your EMI is very comfortable compared to income."
        elif emi_to_income_pct < 45:
            affordability_msg = "Moderate. Manageable but keep expenses controlled."
        else:
            affordability_msg = "High EMI burden. Consider reducing loan amount or increasing tenure."

        # Prepayment impact
        prepay_interest_saved = 0.0
        prepay_months_saved = 0
        prepay_new_tenure_years = tenure_years

        if has_prepay and prepay_month <= base_months:
            schedule_prepay, total_interest_prepay, _ = create_amortization_with_prepay(
                loan_amount, interest_rate, tenure_years, emi, prepay_amount, prepay_month
            )
            new_months = len(schedule_prepay)
            prepay_interest_saved = total_interest - total_interest_prepay
            prepay_months_saved = base_months - new_months
            prepay_new_tenure_years = new_months / 12.0

        credit_band, credit_comment, credit_tips = get_credit_insights(credit_score)

        return render_template(
            "result.html",
            current_page="home",
            emi=round(emi, 2),
            approval_prob=round(approval_prob, 2),
            risk_level=risk_level,
            risk_score=round(risk_score, 2),
            emi_to_income_pct=round(emi_to_income_pct, 2),
            affordability_msg=affordability_msg,
            total_interest=round(total_interest, 2),
            total_principal=round(total_principal, 2),
            total_payment=round(total_payment, 2),
            schedule=schedule,
            balance_labels_json=balance_labels_json,
            balance_data_json=balance_data_json,
            loan_amount=loan_amount,
            interest_rate=interest_rate,
            tenure_years=tenure_years,
            has_prepay=has_prepay,
            prepay_amount=prepay_amount,
            prepay_month=prepay_month,
            prepay_interest_saved=round(prepay_interest_saved, 2),
            prepay_months_saved=prepay_months_saved,
            prepay_new_tenure_years=round(prepay_new_tenure_years, 2),
            credit_score=int(credit_score),
            credit_band=credit_band,
            credit_comment=credit_comment,
            credit_tips=credit_tips
        )

    except Exception as e:
        return f"<h3>Error: {e}</h3>"


# ---------- NEW STANDALONE TOOL ROUTES ----------

@app.route("/credit-advisor", methods=["GET", "POST"])
def credit_advisor():
    """
    Standalone Credit Score Advisor tool.
    """
    result = None

    if request.method == "POST":
        try:
            score = float(request.form["credit_score"])
            band, comment, tips = get_credit_insights(score)
            result = {
                "score": int(score),
                "band": band,
                "comment": comment,
                "tips": tips
            }
        except Exception as e:
            result = {"error": str(e)}

    return render_template("credit_advisor.html",
                           current_page="home",
                           result=result)


@app.route("/prepayment-analyzer", methods=["GET", "POST"])
def prepayment_analyzer():
    """
    Standalone prepayment analyzer.
    """
    result = None

    if request.method == "POST":
        try:
            loan_amount = float(request.form["loan_amount"])
            interest_rate = float(request.form["interest_rate"])
            tenure_years = float(request.form["tenure_years"])
            prepay_amount = float(request.form["prepay_amount"])
            prepay_month = int(request.form["prepay_month"])

            emi = calc_emi(loan_amount, interest_rate, tenure_years)
            base_sched, base_interest, base_principal = create_amortization_schedule(
                loan_amount, interest_rate, tenure_years, emi
            )
            base_months = len(base_sched)

            pre_sched, pre_interest, _ = create_amortization_with_prepay(
                loan_amount, interest_rate, tenure_years, emi,
                prepay_amount, prepay_month
            )
            new_months = len(pre_sched)

            interest_saved = base_interest - pre_interest
            months_saved = base_months - new_months
            new_tenure_years = new_months / 12.0

            result = {
                "emi": round(emi, 2),
                "base_interest": round(base_interest, 2),
                "new_interest": round(pre_interest, 2),
                "interest_saved": round(interest_saved, 2),
                "base_months": base_months,
                "new_months": new_months,
                "months_saved": months_saved,
                "new_tenure_years": round(new_tenure_years, 2)
            }
        except Exception as e:
            result = {"error": str(e)}

    return render_template("prepayment_analyzer.html",
                           current_page="home",
                           result=result)


@app.route("/risk-eligibility", methods=["GET", "POST"])
def risk_eligibility():
    """
    Standalone Risk & Eligibility tool.
    """
    result = None

    if request.method == "POST":
        try:
            monthly_income = float(request.form["monthly_income"])
            existing_emi = float(request.form["existing_emi"])
            loan_amount = float(request.form["loan_amount"])
            tenure_years = float(request.form["tenure_years"])
            interest_rate = float(request.form["interest_rate"])
            credit_score = float(request.form["credit_score"])

            emi = calc_emi(loan_amount, interest_rate, tenure_years)

            risk_score = (
                0.4 * (existing_emi / (monthly_income + 1)) * 100 +
                0.3 * (loan_amount / (monthly_income * 12 + 1)) * 10 -
                0.3 * ((credit_score - 550) / 300) * 100
            )

            X_app = np.array([[
                monthly_income, existing_emi, loan_amount,
                tenure_years, credit_score, risk_score
            ]])
            approval_prob = approval_model.predict_proba(X_app)[0][1] * 100

            emi_to_income_pct = (emi / monthly_income) * 100 if monthly_income > 0 else 0

            if risk_score < 30:
                risk_level = "Low"
            elif risk_score < 60:
                risk_level = "Medium"
            else:
                risk_level = "High"

            result = {
                "emi": round(emi, 2),
                "risk_score": round(risk_score, 2),
                "risk_level": risk_level,
                "approval_prob": round(approval_prob, 2),
                "emi_to_income_pct": round(emi_to_income_pct, 2)
            }
        except Exception as e:
            result = {"error": str(e)}

    return render_template("risk_eligibility.html",
                           current_page="home",
                           result=result)


@app.route("/emi-simulator", methods=["GET", "POST"])
def emi_simulator():
    """
    Standalone EMI Simulator – compare base vs. alternate rate/tenure.
    """
    result = None

    if request.method == "POST":
        try:
            loan_amount = float(request.form["loan_amount"])
            interest_rate = float(request.form["interest_rate"])
            tenure_years = float(request.form["tenure_years"])

            alt_rate_str = request.form.get("alt_interest_rate", "").strip()
            alt_tenure_str = request.form.get("alt_tenure_years", "").strip()

            emi_base = calc_emi(loan_amount, interest_rate, tenure_years)

            alt_rate = interest_rate if not alt_rate_str else float(alt_rate_str)
            alt_tenure = tenure_years if not alt_tenure_str else float(alt_tenure_str)

            emi_alt = calc_emi(loan_amount, alt_rate, alt_tenure)

            result = {
                "emi_base": round(emi_base, 2),
                "rate_base": interest_rate,
                "tenure_base": tenure_years,
                "emi_alt": round(emi_alt, 2),
                "rate_alt": alt_rate,
                "tenure_alt": alt_tenure
            }
        except Exception as e:
            result = {"error": str(e)}

    return render_template("emi_simulator.html",
                           current_page="home",
                           result=result)


# ---------- Download routes ----------

@app.route("/download_pdf")
def download_pdf():
    loan_amount = float(request.args.get("loan_amount", 0))
    interest_rate = float(request.args.get("interest_rate", 0))
    tenure_years = float(request.args.get("tenure_years", 0))
    emi = float(request.args.get("emi", 0))
    total_interest = float(request.args.get("total_interest", 0))
    total_payment = float(request.args.get("total_payment", 0))
    approval_prob = float(request.args.get("approval_prob", 0))
    credit_score = request.args.get("credit_score", "N/A")
    credit_band = request.args.get("credit_band", "N/A")
    risk_level = request.args.get("risk_level", "N/A")
    emi_pct = float(request.args.get("emi_to_income_pct", 0))

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    y = h - 60

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "Smart Financial Advisor - EMI Report")
    y -= 30
    c.setFont("Helvetica", 11)
    c.drawString(50, y, "Personalised EMI, risk & credit analysis")
    y -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Loan Details")
    y -= 18
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Loan Amount: ₹ {loan_amount:,.2f}")
    y -= 14
    c.drawString(50, y, f"Interest Rate: {interest_rate:.2f}% p.a.")
    y -= 14
    c.drawString(50, y, f"Tenure: {tenure_years:.2f} years")

    y -= 22
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "EMI & Cost Summary")
    y -= 18
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Monthly EMI: ₹ {emi:,.2f}")
    y -= 14
    c.drawString(50, y, f"Total Interest: ₹ {total_interest:,.2f}")
    y -= 14
    c.drawString(50, y, f"Total Repayment: ₹ {total_payment:,.2f}")
    y -= 14
    c.drawString(50, y, f"EMI as % of Income: {emi_pct:.2f}%")

    y -= 22
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Approval & Risk")
    y -= 18
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Approval Probability: {approval_prob:.2f}%")
    y -= 14
    c.drawString(50, y, f"Risk Level: {risk_level}")

    y -= 22
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Credit Profile")
    y -= 18
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Credit Score: {credit_score} ({credit_band})")

    y -= 30
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y, "Note: Demo report for learning only; real banks use more parameters.")

    c.showPage()
    c.save()
    buf.seek(0)

    return send_file(buf, as_attachment=True,
                     download_name="emi_report.pdf",
                     mimetype="application/pdf")


@app.route("/download_schedule")
def download_schedule():
    loan_amount = float(request.args.get("loan_amount", 0))
    interest_rate = float(request.args.get("interest_rate", 0))
    tenure_years = float(request.args.get("tenure_years", 0))

    emi = calc_emi(loan_amount, interest_rate, tenure_years)
    schedule, _, _ = create_amortization_schedule(loan_amount, interest_rate, tenure_years, emi)

    s_buf = io.StringIO()
    writer = csv.writer(s_buf)
    writer.writerow(["Month", "EMI", "Interest", "Principal", "Balance"])
    for row in schedule:
        writer.writerow([row["month"], row["emi"], row["interest"], row["principal"], row["balance"]])

    b_buf = io.BytesIO(s_buf.getvalue().encode("utf-8"))
    b_buf.seek(0)
    return send_file(b_buf, as_attachment=True,
                     download_name="amortization_schedule.csv",
                     mimetype="text/csv")


# ---------- Chatbot route ----------

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.form.get("user_msg", "").lower()

    if "emi" in msg and "how" in msg:
        reply = "Use the EMI form on the left: enter loan amount, interest rate and tenure."
    elif "emi" in msg:
        reply = "Try to keep total EMIs under ~40% of your monthly income."
    elif "credit score" in msg:
        reply = "A score above 750 is good; above 800 is excellent."
    elif "improve" in msg and "score" in msg:
        reply = "Pay EMIs and bills on time, reduce card balances, and avoid many new loans."
    elif "prepay" in msg or "prepayment" in msg:
        reply = "Prepaying part of the loan can reduce total interest and shorten the tenure."
    elif "compare" in msg:
        reply = "Use the Compare Loans tab in the top menu to compare two offers."
    else:
        reply = "You can ask about EMI, credit score, risk, prepayment, or loan affordability."

    return reply


# ---------- Main ----------

if __name__ == "__main__":
    app.run(debug=True)
