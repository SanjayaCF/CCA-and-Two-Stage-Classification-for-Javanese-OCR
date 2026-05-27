"""
Pilih 50 gambar baris aksara Jawa secara acak dari folder luaran,
terdistribusi merata antar dokumen, salin ke static/test_images/.
"""
import os, random, re, shutil
from pathlib import Path

LUARAN = Path(r"C:\Users\SanjayaCF\Desktop\Skripsi\Kerja Praktik\Y2outputs\luaran")
OUTPUT = Path(__file__).parent / "static" / "test_images"
N_TOTAL = 50
SEED    = 42

def main():
    random.seed(SEED)

    # Kelompokkan per dokumen
    docs = {}
    for f in sorted(LUARAN.glob("*.jpg")):
        doc = f.stem.split(" ")[0]
        docs.setdefault(doc, []).append(f)

    doc_ids = sorted(docs.keys())
    print(f"Dokumen ditemukan: {len(doc_ids)}  |  Total baris: {sum(len(v) for v in docs.values())}")

    # 1 sampel per dokumen (39 gambar)
    selected = [random.choice(docs[d]) for d in doc_ids]

    # Tambah 11 sampel kedua dari dokumen acak (agar total = 50)
    extra_docs = random.sample(doc_ids, N_TOTAL - len(doc_ids))
    for d in extra_docs:
        remaining = [f for f in docs[d] if f not in selected]
        if remaining:
            selected.append(random.choice(remaining))

    selected = selected[:N_TOTAL]

    # Bersihkan output & salin
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for f in OUTPUT.glob("*.jpg"):
        f.unlink()

    for src in selected:
        dst_name = re.sub(r"\s+", "_", src.name)
        shutil.copy(src, OUTPUT / dst_name)

    print(f"Disalin {len(selected)} gambar ke: {OUTPUT}")
    for f in sorted(OUTPUT.glob("*.jpg")):
        print(f"  {f.name}")

if __name__ == "__main__":
    main()
