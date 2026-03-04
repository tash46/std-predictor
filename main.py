from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
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

        #required_columns = ["Mean", "A", "B", "C", "D", "E"]

        # Validate columns
        if not all(col in df.columns for col in feature_cols):
            return {"error": f"Excel must contain columns: {feature_cols}"}

        X = df[feature_cols]

        # Predict
        predictions = model.predict(X)
        df["Predicted_STD"] = predictions

        # Save to temporary file (safe for Render)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            output_path = tmp.name

        df.to_excel(output_path, index=False)

        return FileResponse(
            path=output_path,
            filename=f"{Path(file.filename).stem}_result.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        print("ERROR OCCURRED:")
        traceback.print_exc()
        return {"error": str(e)}