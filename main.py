from fastapi import Form, File, FastAPI
from fastapi.responses import FileResponse, Response
from re import sub
from grading import grade

app: FastAPI = FastAPI()

@app.get("/")
def root():
    return FileResponse("index.html")

@app.get("/template")
def root():
    return FileResponse("template.jpg")

@app.post("/grade")
def submit(hl_correct_answers = Form(...), sl_correct_answers = Form(...), file = File(...)):
    return Response(content=grade(file.file.read(), [list(sub(r'[^A-D]', '', hl_correct_answers.upper())), list(sub(r'[^A-D]', '', sl_correct_answers.upper()))]), media_type="application/pdf", headers={"Content-Disposition": "inline; filename=output.pdf"})