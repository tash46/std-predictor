from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from io import BytesIO
import tempfile
import traceback

app = FastAPI()

# Load model once
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "std_predictor_v2_0303_2.joblib"

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found at {model_path}")

bundle = joblib.load(model_path)

# Extract actual model
model = bundle["model"]

# Extract expected feature order
feature_cols = bundle["features"]

# -----------------------------
# Feature Engineering
# -----------------------------
def engineer_features(df):
    """
    Must match training logic exactly.
    """

    # Cumulative ratios
    df['cum_A'] = df['A']
    df['cum_B'] = df['A'] + df['B']
    df['cum_C'] = df['A'] + df['B'] + df['C']
    df['cum_D'] = df['A'] + df['B'] + df['C'] + df['D']

    # Entropy
    ratios = ["A", "B", "C", "D", "E"]
    df['entropy'] = df[ratios].apply(
        lambda x: -np.sum(x * np.log(x + 1e-9)), axis=1
    )

    # Zero bins
    df['zero_bins'] = (df[ratios] == 0).sum(axis=1)

    return df

@app.get("/", response_class=HTMLResponse)
def upload_page():
    return """
    <html>
        <body style="font-family: Arial; text-align: center; margin-top: 50px;">
            <h2>Standard Deviation Predictor</h2>
            <form action="/predict/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept=".xlsx" required>
                <br><br>
                <input type="submit" value="Upload and Predict">
            </form>
        </body>
    </html>
    """

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded file safely (Fix for Render issue)
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents), engine="openpyxl")

        required_columns = ["mean", "A", "B", "C", "D", "E"]

        # Validate columns
        if not all(col in df.columns for col in required_columns):
            return {"error": f"Excel must contain columns: {required_columns}"}
        
        df = engineer_features(df)

        X = df[feature_cols].values

        # Predict
        predictions = model.predict(X)
        #df["Predicted_STD"] = predictions
        output_df = df[["mean", "A", "B", "C", "D", "E"]].copy()
        output_df["Predicted_STD"] = predictions

        # Save to temporary file (safe for Render)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            output_path = tmp.name

        output_df.to_excel(output_path, index=False)

        return FileResponse(
            path=output_path,
            filename=f"{Path(file.filename).stem}_result.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        print("ERROR OCCURRED:")
        traceback.print_exc()
        return {"error": str(e)}