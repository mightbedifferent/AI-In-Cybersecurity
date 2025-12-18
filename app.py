from flask import Flask, request, render_template_string
import joblib
import numpy as np
import os
import importlib.util

# -----------------------------
# Configuration (make sure these files exist in the same folder)
# -----------------------------
DETECTOR_FILE = "Phishing_Det.py"          # your main python file name (with space)
MODEL_FILE = "phishing_model.joblib"       # trained model


# -----------------------------
# Load your detector module dynamically (works even if filename has spaces)
# -----------------------------
def load_detector_module(path: str):
    spec = importlib.util.spec_from_file_location("phishing_detector_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
detector_path = os.path.join(BASE_DIR, DETECTOR_FILE)
model_path = os.path.join(BASE_DIR, MODEL_FILE)

detector = load_detector_module(detector_path)
model = joblib.load(model_path)

app = Flask(__name__)

# -----------------------------
# Stylish Dark UI + Reason
# -----------------------------
HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>AI Phishing Detection</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

*{ box-sizing:border-box; }
body{
  margin:0;
  min-height:100vh;
  background: radial-gradient(1200px 600px at 20% 10%, #2b2c31 0%, transparent 50%),
              radial-gradient(900px 500px at 90% 30%, #1f2230 0%, transparent 55%),
              linear-gradient(135deg, #0b0c10 0%, #15161b 45%, #0b0c10 100%);
  font-family:'Inter', sans-serif;
  color:#eaeaea;
  display:flex;
  align-items:center;
  justify-content:center;
  padding:24px;
}

.container{ width:100%; max-width:920px; }
.card{
  background: rgba(16, 17, 22, 0.92);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  padding: 34px;
  box-shadow: 0 25px 60px rgba(0,0,0,0.55);
  backdrop-filter: blur(8px);
}

h1{
  margin:0;
  text-align:center;
  font-weight:600;
  letter-spacing:0.3px;
}
.subtitle{
  text-align:center;
  color:#a7a7a7;
  margin-top:10px;
  margin-bottom:28px;
  font-size:14px;
}

label{ font-size:13px; color:#b7b7b7; }
textarea, input, select{
  width:100%;
  background:#0c0d12;
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 12px;
  padding: 14px;
  color:#eaeaea;
  font-size:14px;
  margin-top:7px;
}
textarea{ height:150px; resize:none; }

textarea:focus, input:focus, select:focus{
  outline:none;
  border-color:#4f7cff;
  box-shadow: 0 0 0 4px rgba(79,124,255,0.18);
}

.grid{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-top: 16px;
}

button{
  margin-top:24px;
  width:100%;
  border:none;
  border-radius:14px;
  padding: 14px;
  font-weight:600;
  font-size:15px;
  color:white;
  cursor:pointer;
  background: linear-gradient(135deg, #4f7cff 0%, #6c5cff 55%, #4f7cff 100%);
  box-shadow: 0 14px 30px rgba(79,124,255,0.25);
}
button:hover{ opacity:0.92; }

.result{
  margin-top:26px;
  background:#0c0d12;
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 20px;
}

.rowline{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
  flex-wrap:wrap;
}

.badge{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding:6px 12px;
  border-radius:999px;
  font-weight:600;
  font-size:13px;
}
.safe{ color:#2ecc71; background: rgba(46,204,113,0.14); }
.phishing{ color:#e74c3c; background: rgba(231,76,60,0.14); }
.suspicious{ color:#f1c40f; background: rgba(241,196,15,0.14); }

small{ color:#9a9a9a; }

.section-title{
  margin-top:14px;
  margin-bottom:8px;
  font-weight:600;
  color:#d8d8d8;
  font-size:13px;
}

.reason{
  color:#d0d0d0;
  line-height:1.65;
  font-size:14px;
  padding: 12px;
  border-radius: 12px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
}

code{
  display:block;
  margin-top:8px;
  background:#07080c;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius:12px;
  padding:12px;
  font-size:12px;
  color:#a9a9a9;
  overflow-x:auto;
}

ul{ margin: 8px 0 0 18px; padding:0; }
li{ margin: 6px 0; color:#cfcfcf; font-size:13px; }

.footer{
  text-align:center;
  margin-top: 18px;
  color:#7e7e7e;
  font-size:12px;
}

@media(max-width:760px){
  .grid{ grid-template-columns:1fr; }
}
</style>
</head>

<body>
<div class="container">
  <div class="card">
    <h1>AI Phishing Detection</h1>
    <div class="subtitle">Local ML Demo · Feature Engineering · Explainable Output</div>

    <form method="post">
      <label>Message Content</label>
      <textarea name="text" placeholder="Paste the message here...">{{text}}</textarea>

      <div class="grid">
        <div>
          <label>Source</label>
          <select name="msg_source">
            {% for s in ["email","sms","social","unknown"] %}
              <option value="{{s}}" {% if msg_source==s %}selected{% endif %}>{{s}}</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>Sender Domain / Name</label>
          <input name="sender" value="{{sender}}" placeholder="e.g., google.com, paypal.com / Name" />
        </div>
      </div>

      <button type="submit">Analyze</button>
    </form>

    {% if result %}
    <div class="result">
      <div class="rowline">
        <div>
          <span class="badge {{result.badge_class}}">
            {{result.pred}}
          </span>
          <small>&nbsp;Confidence: <b>{{result.conf}}</b></small>
        </div>
        <small>Probabilities: {{result.probs}}</small>
      </div>

      <div class="section-title">Reason (Why the AI decided this)</div>
      <div class="reason">{{result.reason}}</div>

      <div class="section-title">Feature Preview</div>
      <code>{{result.preview}}</code>

      <div class="section-title">Top Features Pushing Toward Phishing</div>
      {% if result.top %}
        <ul>
          {% for name, score in result.top %}
            <li>{{name}} → {{ "%.4f"|format(score) }}</li>
          {% endfor %}
        </ul>
      {% else %}
        <small>No strong phishing-driving features found.</small>
      {% endif %}
    </div>
    {% endif %}

  </div>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    msg_source = "email"
    sender = ""
    result = None

    if request.method == "POST":
        text = request.form.get("text", "")
        msg_source = request.form.get("msg_source", "unknown")
        sender = request.form.get("sender", "unknown")

        # Use your own detector code predict_one(model, text, source, sender)
        out = detector.predict_one(model, text=text, source=msg_source, sender=sender)

        pred = str(out.get("prediction", "unknown")).upper()
        conf = out.get("confidence", 0.0)
        probs = out.get("probabilities", {})
        top = out.get("top_phishing_features", [])
        preview = out.get("feature_preview", "")

        # Reason (fallback if not present)
        reason = out.get("reason")
        if not reason:
            reason = "No explanation string was returned. (Make sure you added generate_reason + returned 'reason' in predict_one.)"

        # badge color
        badge_class = "safe"
        if pred == "PHISHING":
            badge_class = "phishing"
        elif pred == "SUSPICIOUS":
            badge_class = "suspicious"

        result = {
            "pred": pred,
            "conf": conf,
            "probs": probs,
            "reason": reason,
            "top": top,
            "preview": preview,
            "badge_class": badge_class,
        }

    return render_template_string(
        HTML,
        text=text,
        msg_source=msg_source,
        sender=sender,
        result=result
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
