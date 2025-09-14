import os
import re
import json
import unicodedata

def clean_legal_text(text):
    text = unicodedata.normalize('NFKC', text)

    text = text.lower()

    lines = text.split('\n')
    clean_lines = []
    boilerplate_patterns = [
        r'^\s*reportable\s*$',
        r'^page\s*\d+\s*of\s*\d+',
        r'^coram:.*$',
        r'.*scanned by camscanner.*'
    ]
    for ln in lines:
        if any(re.match(pat, ln.strip(), flags=re.IGNORECASE) for pat in boilerplate_patterns):
            continue
        clean_lines.append(ln)

    text = '\n'.join(clean_lines)

    text = re.sub('\t', ' ', text)
    text = re.sub(r'\s+\n', '\n', text)
    text = re.sub(r'\n\s+', '\n', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{3,}', '.', text)
    test = re.sub(r'(- ){3,}', '', text)

    text = text.replace('“','"').replace('”','"').replace('’',"'").replace('–','-').replace('…', '...')
    text = re.sub(r'-\s+', '', text) 

    return text.strip()

unwanted_line = "Take notes as you read a judgment using our Virtual Legal Assistant and get email alerts whenever a new judgment matches your query (Query Alert Service). Try out our Premium Member Services -- Sign up today and get free trial for one month."

def batch_clean_text_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.endswith('.txt'):
            continue
        with open(os.path.join(input_dir, fname), encoding='utf-8') as f:
            original = f.read()
        if original.startswith(unwanted_line):
            original = original[len(unwanted_line):].lstrip()
        cleaned = clean_legal_text(original)
        output_json = {
            "case_file": fname,
            "original_text": original,
            "cleaned_text": cleaned
        }
        outname = os.path.splitext(fname)[0] + ".json"
        with open(os.path.join(output_dir, outname), 'w', encoding='utf-8') as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)
        print(f"Processed {fname} -> {outname}")

batch_clean_text_files('/kaggle/input/indian-court-judgments-raw-text-corpus', './cleaned_cases/')
