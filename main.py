from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
import joblib
import os
from pathlib import Path

app = FastAPI()

# Load model once
model = joblib.load("std_predictor_v2_0303_2.joblib")

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

    # Read Excel
    df = pd.read_excel(file.file)

    required_columns = ["Mean", "A", "B", "C", "D", "E"]

    # Validate columns
    if not all(col in df.columns for col in required_columns):
        return {"error": "Excel must contain columns: Mean, A, B, C, D, E"}

    X = df[required_columns]

    # Predict
    predictions = model.predict(X)

    df["Predicted_STD"] = predictions

    # Create output filename
    original_name = Path(file.filename).stem
    output_filename = f"{original_name}_result.xlsx"
    output_path = output_filename

    df.to_excel(output_path, index=False)

    return FileResponse(
        path=output_path,
        filename=output_filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )