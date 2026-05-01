import os
from pypdf import PdfReader

def extract_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading {pdf_path}: {e}"

papers_dir = "/home/eran.b/takehome/papers"
output_dir = "/home/eran.b/takehome/scratch/paper_texts"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(papers_dir):
    if filename.endswith(".pdf"):
        full_path = os.path.join(papers_dir, filename)
        print(f"Extracting {filename}...")
        text = extract_text(full_path)
        with open(os.path.join(output_dir, filename.replace(".pdf", ".txt")), "w") as f:
            f.write(text)
print("Extraction complete.")
