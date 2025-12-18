import re
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# 1) Data structure
# ============================================================
@dataclass
class Sample:
    text: str
    source: str   # email / sms / social / unknown
    sender: str   # domain or name
    label: str    # phishing / safe


# ============================================================
# 2) Brand impersonation / typosquatting detection (IMPORTANT)
# ============================================================
BRANDS = {
    "google": "google",
    "paypal": "paypal",
    "microsoft": "microsoft",
    "apple": "apple",
    "amazon": "amazon",
    "github": "github",
    "netflix": "netflix",
    "facebook": "facebook",
}

OFFICIAL_DOMAINS = {
    "google.com", "paypal.com", "microsoft.com", "apple.com", "amazon.com",
    "github.com", "netflix.com", "facebook.com"
}

def extract_domain(sender: str) -> str:
    s = (sender or "").strip().lower()
    return s.split("@", 1)[1] if "@" in s else s

def tld_of(domain: str) -> str:
    return domain.rsplit(".", 1)[-1].lower() if "." in domain else ""

def root_of(domain: str) -> str:
    d = extract_domain(domain)
    return d.split(".", 1)[0].lower() if "." in d else d.lower()

def levenshtein(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                cur[j - 1] + 1,            # insert
                prev[j] + 1,               # delete
                prev[j - 1] + (ca != cb)   # substitute
            ))
        prev = cur
    return prev[-1]

def brand_impersonation(domain: str) -> Optional[str]:
    """
    Returns brand name if domain looks like a typo of a known brand.
    Example: gogle.com -> "google"
    """
    d = extract_domain(domain)

    # exact official domains -> not impersonation
    if d in OFFICIAL_DOMAINS:
        return None

    r = root_of(d)
    if len(r) < 4:
        return None

    best_brand = None
    best_dist = 999

    for brand, official_root in BRANDS.items():
        dist = levenshtein(r, official_root)
        if dist < best_dist:
            best_dist = dist
            best_brand = brand

    # Threshold: distance 1 or 2 catches most typos (gogle/google, paypol/paypal, etc.)
    if best_brand and best_dist <= 2 and abs(len(r) - len(BRANDS[best_brand])) <= 2:
        return best_brand

    return None


# ============================================================
# 3) Demo dataset (MORE SAMPLES)
# ============================================================
def demo_dataset() -> List[Sample]:
    phishing_email = [
        ("Urgent: verify your account immediately to avoid suspension.", "email", "secure-login-alert.com"),
        ("Security alert: unusual login detected. Confirm your identity now.", "email", "security-team-verify.xyz"),
        ("Your mailbox storage is full. Click the link to upgrade.", "email", "mail-upgrade-support.top"),
        ("Your payment failed. Update your billing information immediately.", "email", "billing-update-now.com"),
        ("Final notice: your account will be locked. Verify now.", "email", "final-notice-secure.live"),
        ("We blocked a sign-in attempt. Verify it was you.", "email", "signin-verify-security.xyz"),
        ("Action required: reset your password to keep access.", "email", "reset-password-support.click"),
        ("Your tax refund is pending. Confirm your details.", "email", "refund-verify-now.top"),
        ("New document shared with you. Sign in to view.", "email", "sharepoint-secure-login.xyz"),
        ("Your subscription will expire today. Update payment details.", "email", "subscription-billing-update.live"),
        ("Important: validate your account information.", "email", "validate-account-portal.top"),
        ("Your account has been compromised. Reset credentials now.", "email", "compromised-reset-credentials.xyz"),
    ]

    phishing_sms = [
        ("Your bank account is on hold. Verify now.", "sms", "unknown"),
        ("You won a prize! Click link to claim now.", "sms", "promo-gifts.live"),
        ("Package pending. Update delivery details now.", "sms", "track-delivery.top"),
        ("Suspicious transaction detected. Verify immediately.", "sms", "card-alert.xyz"),
        ("Payment failed. Update info now.", "sms", "billing-update-now.com"),
        ("Delivery failed. Pay a small fee to reschedule.", "sms", "delivery-fee.top"),
        ("Urgent: verify your account to continue service.", "sms", "service-verify.live"),
    ]

    phishing_social = [
        ("Is this you in this video? Click here!", "social", "t.co"),
        ("Hey check this link urgently, I think you were tagged.", "social", "short.link"),
        ("Your account will be disabled. Act now.", "social", "support-verify.xyz"),
        ("You received a secure message. Login to view.", "social", "secure-msg.top"),
        ("Claim your gift card today! Limited time.", "social", "promo-gifts.live"),
    ]

    # Typosquatting examples (so the ML learns it too)
    typo_phishing = [
        ("Google security notice: verify your account now.", "email", "gogle.com"),
        ("PayPal alert: unusual transaction detected. Confirm now.", "email", "paypol.com"),
        ("Microsoft account warning: reset your password now.", "email", "micros0ft-support.xyz"),
        ("Apple ID locked. Verify to restore access.", "email", "applle.com"),
        ("GitHub Security: unusual login attempt. Verify now.", "email", "githab.com"),
        ("Netflix billing issue. Update payment details now.", "email", "netflx-billing.top"),
    ]

    safe_email = [
        ("Meeting at 3 PM today, please confirm.", "email", "company.com"),
        ("Please review the attached document and share feedback.", "email", "company.com"),
        ("Here is the report you requested. Let me know if you have questions.", "email", "company.com"),
        ("Invoice attached for last month. Thank you.", "email", "vendor.com"),
        ("Reminder: project deadline is next week.", "email", "company.com"),
        ("Can we reschedule our meeting to tomorrow?", "email", "company.com"),
        ("Thanks for your help earlier.", "email", "company.com"),
        ("Your reservation is confirmed.", "email", "hotel.com"),
        ("Please approve the PR when you have time.", "email", "github.com"),
        ("Google calendar invite: Team sync tomorrow.", "email", "google.com"),
        ("PayPal receipt for your recent purchase.", "email", "paypal.com"),
        ("Microsoft Teams meeting link for today.", "email", "microsoft.com"),
    ]

    safe_sms = [
        ("Are we still meeting today?", "sms", "friend"),
        ("I will call you later.", "sms", "friend"),
        ("Running a bit late, see you soon.", "sms", "friend"),
        ("Thanks!", "sms", "friend"),
        ("Send me the file when you can.", "sms", "friend"),
    ]

    safe_social = [
        ("Happy birthday!", "social", "instagram.com"),
        ("Let’s catch up soon.", "social", "instagram.com"),
        ("Thanks for the update.", "social", "linkedin.com"),
        ("See you tomorrow.", "social", "whatsapp.com"),
    ]

    data: List[Sample] = []
    for t, s, snd in phishing_email + phishing_sms + phishing_social + typo_phishing:
        data.append(Sample(text=t, source=s, sender=snd, label="phishing"))
    for t, s, snd in safe_email + safe_sms + safe_social:
        data.append(Sample(text=t, source=s, sender=snd, label="safe"))

    return data


# ============================================================
# 4) Feature engineering
# ============================================================
SUSPICIOUS_TLDS = {"xyz", "top", "click", "live", "icu", "biz"}
SUSPICIOUS_DOMAIN_WORDS = {"login", "verify", "secure", "update", "billing", "password", "account", "confirm", "support"}
URGENCY_WORDS = {"urgent", "immediately", "now", "asap", "final", "notice", "important", "action required"}
MONEY_WORDS = {"bank", "payment", "card", "invoice", "fee", "prize", "gift", "giveaway", "wallet", "refund", "billing"}
CRED_WORDS = {"password", "login", "verify", "account", "confirm", "credentials", "identity"}

URL_RE = re.compile(r"(https?://[^\s]+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)

def count_urls(text: str) -> int:
    return len(URL_RE.findall(text or ""))

def contains_any(text: str, vocab: set) -> bool:
    t = (text or "").lower()
    return any(w in t for w in vocab)

def has_digits(s: str) -> bool:
    return any(ch.isdigit() for ch in (s or ""))

def build_feature_text(sample: Sample) -> str:
    text = (sample.text or "").strip().lower()
    src = (sample.source or "unknown").strip().lower()
    sender = (sample.sender or "unknown").strip().lower()

    domain = extract_domain(sender)
    tld = tld_of(domain)

    tokens = [f"SRC_{src}"]

    # Brand / typosquatting features
    if domain in OFFICIAL_DOMAINS:
        tokens.append("DOMAIN_OFFICIAL_BRAND")

    brand = brand_impersonation(domain)
    if brand:
        tokens.append("DOMAIN_TYPO_SUSPECT")
        tokens.append(f"BRAND_IMPERSONATION_{brand}")

    # Sender tokens
    if sender in {"unknown", ""}:
        tokens.append("SENDER_UNKNOWN")
    else:
        tokens.append("SENDER_KNOWN")

    # Domain shape
    if has_digits(domain):
        tokens.append("DOMAIN_HAS_DIGITS")
    if "-" in domain:
        tokens.append("DOMAIN_HAS_HYPHEN")

    # TLD suspicion
    if tld in SUSPICIOUS_TLDS:
        tokens.append("TLD_SUSPICIOUS")
    elif tld:
        tokens.append("TLD_NORMAL")

    # Suspicious words in domain
    if any(w in domain for w in SUSPICIOUS_DOMAIN_WORDS):
        tokens.append("DOMAIN_SUSPICIOUS_WORD")

    # Content tokens
    u = count_urls(text)
    tokens.append("NO_URL" if u == 0 else "ONE_URL" if u == 1 else "MULTI_URL")

    if "http://" in text:
        tokens.append("HTTP_LINK")

    if contains_any(text, URGENCY_WORDS):
        tokens.append("HAS_URGENCY")
    if contains_any(text, MONEY_WORDS):
        tokens.append("HAS_MONEY_THEME")
    if contains_any(text, CRED_WORDS):
        tokens.append("ASKS_CREDENTIALS")

    if "click" in text or "link" in text:
        tokens.append("CALL_TO_CLICK")

    if EMAIL_RE.search(text):
        tokens.append("CONTAINS_EMAIL_ADDRESS")

    if text.count("!") >= 2:
        tokens.append("MANY_EXCLAMATIONS")

    if len(text) < 30:
        tokens.append("SHORT_MESSAGE")
    elif len(text) > 160:
        tokens.append("LONG_MESSAGE")

    return text + " " + " ".join(tokens)


# ============================================================
# 5) ML pipeline
# ============================================================
def make_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        ))
    ])

def prepare_xy(samples: List[Sample]) -> Tuple[List[str], List[str]]:
    X = [build_feature_text(s) for s in samples]
    y = [s.label for s in samples]
    return X, y


# ============================================================
# 6) Train / Evaluate
# ============================================================
def train_model(samples: List[Sample], seed: int = 42) -> Tuple[Pipeline, Dict]:
    X, y = prepare_xy(samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    model = make_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)
    labels_order = ["phishing", "safe"]
    cm = confusion_matrix(y_test, y_pred, labels=labels_order).tolist()

    metrics = {
        "accuracy": float(acc),
        "report": report,
        "confusion_matrix": cm,
        "labels_order": labels_order,
        "test_preview": list(zip(X_test[:5], y_test[:5], y_pred[:5]))
    }
    return model, metrics

def cross_validate(samples: List[Sample], folds: int = 5) -> float:
    X, y = prepare_xy(samples)
    model = make_pipeline()
    scores = cross_val_score(model, X, y, cv=folds, scoring="accuracy")
    return float(scores.mean())


# ============================================================
# 7) Explainability (fixed for binary LR)
# ============================================================
def explain_top_features_toward_phishing(model: Pipeline, feature_text: str, top_k: int = 10):
    tfidf: TfidfVectorizer = model.named_steps["tfidf"]
    clf: LogisticRegression = model.named_steps["clf"]

    vec = tfidf.transform([feature_text]).toarray().ravel()
    names = np.array(tfidf.get_feature_names_out())

    weights = clf.coef_[0]          # binary: (1, n_features)
    positive_class = clf.classes_[1] # weights point toward classes_[1]
    contrib = vec * weights

    results = []
    if positive_class == "phishing":
        idxs = np.argsort(contrib)[::-1]
        for i in idxs:
            if vec[i] > 0 and contrib[i] > 0:
                results.append((names[i], float(contrib[i])))
            if len(results) >= top_k:
                break
    else:
        idxs = np.argsort(contrib)  # most negative first
        for i in idxs:
            if vec[i] > 0 and contrib[i] < 0:
                results.append((names[i], float(-contrib[i])))
            if len(results) >= top_k:
                break

    return results

def generate_reason(prediction: str, confidence: float, top_features: list, sender: str) -> str:
    reasons = []

    domain = extract_domain(sender)
    brand = brand_impersonation(domain)

    if prediction == "phishing":
        if brand:
            reasons.append(f"The sender domain looks like an impersonation of a known brand ({brand}).")

        for name, _ in top_features:
            if "URGENT" in name:
                reasons.append("The message contains urgency language.")
            if "ASKS_CREDENTIALS" in name:
                reasons.append("The message asks for sensitive credentials.")
            if "HAS_MONEY_THEME" in name:
                reasons.append("The message is related to money or payments.")
            if "TLD_SUSPICIOUS" in name:
                reasons.append("The domain uses a suspicious top-level domain.")
            if "DOMAIN_TYPO_SUSPECT" in name:
                reasons.append("The domain name appears to be a typo of a legitimate brand.")

    else:  # SAFE
        if confidence > 0.75:
            reasons.append("The message does not show strong phishing indicators.")
        if "DOMAIN_OFFICIAL_BRAND" in " ".join([f for f, _ in top_features]):
            reasons.append("The sender domain matches an official known domain.")
        if not reasons:
            reasons.append("The message structure looks similar to normal legitimate communication.")

    # fallback
    if not reasons:
        reasons.append("The AI model did not find strong indicators of phishing.")

    return " ".join(reasons)

# ============================================================
# 8) Predict
# ============================================================
def predict_one(model: Pipeline, text: str, source: str, sender: str) -> Dict:
    s = Sample(text=text, source=source, sender=sender, label="unknown")
    feat = build_feature_text(s)

    pred = model.predict([feat])[0]
    proba = model.predict_proba([feat])[0]
    conf = float(np.max(proba))
    probs = {cls: float(p) for cls, p in zip(model.named_steps["clf"].classes_, proba)}

    # OPTIONAL security override (stronger): if domain looks like typo of a brand -> force phishing
    domain = extract_domain(sender)
    brand = brand_impersonation(domain)
    if brand and domain not in OFFICIAL_DOMAINS:
        pred = "phishing"
        conf = max(conf, 0.90)
        probs["phishing"] = max(probs.get("phishing", 0.0), 0.90)

    top = explain_top_features_toward_phishing(model, feat, top_k=10)
    reason = generate_reason(pred, conf, top, sender)

    return {
        "prediction": pred,
        "confidence": round(conf, 3),
        "probabilities": {k: round(v, 3) for k, v in probs.items()},
        "top_phishing_features": top,
        "reason": reason,
        "feature_preview": feat[:220] + ("..." if len(feat) > 220 else "")
    }


# ============================================================
# 9) Save/Load
# ============================================================
def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)

def load_model(path: str) -> Pipeline:
    return joblib.load(path)


# ============================================================
# 10) CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="AI Phishing Detection (SAFE vs PHISHING) + Typosquatting")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train and evaluate model, then save it")
    p_train.add_argument("--model-out", default="phishing_model.joblib")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--cv", type=int, default=0, help="Optional CV folds (e.g., 5)")

    p_pred = sub.add_parser("predict", help="Load model and predict (interactive)")
    p_pred.add_argument("--model", default="phishing_model.joblib")
    p_pred.add_argument("--interactive", action="store_true")

    args = parser.parse_args()

    if args.cmd == "train":
        data = demo_dataset()
        model, metrics = train_model(data, seed=args.seed)

        if args.cv and args.cv >= 2:
            metrics["cross_val_accuracy_mean"] = cross_validate(data, folds=args.cv)

        save_model(model, args.model_out)

        print("\n=== Training Finished ===")
        print(f"Saved model to: {args.model_out}")
        print(f"Accuracy (holdout test): {metrics['accuracy']:.3f}")
        if "cross_val_accuracy_mean" in metrics:
            print(f"Cross-val mean accuracy: {metrics['cross_val_accuracy_mean']:.3f}")

        rep = metrics["report"]
        print("\nClassification Report (test split):")
        for lbl in ["phishing", "safe"]:
            if lbl in rep:
                print(f"- {lbl}: precision={rep[lbl]['precision']:.2f}, recall={rep[lbl]['recall']:.2f}, f1={rep[lbl]['f1-score']:.2f}")

        print("\nConfusion Matrix [phishing, safe] rows=true cols=pred:")
        print(metrics["confusion_matrix"])

        print("\nSample test cases (preview):")
        for feat, true_y, pred_y in metrics["test_preview"]:
            print(f"  true={true_y:8s} pred={pred_y:8s}  text='{feat[:90]}...'")

    elif args.cmd == "predict":
        model = load_model(args.model)

        if not args.interactive:
            print("Use: python \"Phishing Det.py\" predict --interactive")
            sys.exit(0)

        print("=== Interactive Mode === (type 'quit' to exit)")
        while True:
            text = input("\nMessage: ").strip()
            if text.lower() == "quit":
                break
            src = input("Source (email/sms/social/unknown): ").strip() or "unknown"
            snd = input("Sender domain/name: ").strip() or "unknown"

            out = predict_one(model, text=text, source=src, sender=snd)

            print("\n=== Prediction ===")
            print("Prediction:", out["prediction"].upper())
            print("Confidence:", out["confidence"])
            print("Probabilities:", out["probabilities"])
            print("Reason:", out["reason"])
            print("Feature preview:", out["feature_preview"])
            print("\nTop features pushing toward PHISHING:")
            if out["top_phishing_features"]:
                for name, score in out["top_phishing_features"]:
                    print(f" - {name:28s} contribution={score:.4f}")
            else:
                print(" - (No strong phishing-driving features found)")


if __name__ == "__main__":
    main()
