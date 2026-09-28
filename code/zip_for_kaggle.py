import zipfile, os

def zip_for_kaggle():
    out_path = "../code_clean_kaggle.zip"
    folders = ["configs", "data", "pykt-toolkit", "src"]
    files = ["requirements.txt", "README.md"]
    
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if os.path.exists(f):
                zf.write(f, arcname=f.replace("\\", "/"))
                
        for d in folders:
            if os.path.exists(d):
                for root, _, fnames in os.walk(d):
                    # Skip massive raw data files
                    if "data\\raw" in root or "data/raw" in root:
                        continue
                    for fname in fnames:
                        full_path = os.path.join(root, fname)
                        arcname = full_path.replace("\\", "/")
                        zf.write(full_path, arcname=arcname)
                        
if __name__ == "__main__":
    zip_for_kaggle()
