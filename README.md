# jsonl-split-id

CLI lokal untuk membagi dataset JSONL menjadi `train.jsonl`, `validation.jsonl`, dan `test.jsonl` dengan seed deterministik. Berguna untuk menyiapkan dataset eksperimen atau fine-tuning AI tanpa mengunggah data ke layanan lain. Split terstratifikasi tersedia untuk menjaga proporsi label per kelas sedekat mungkin saat jumlah record harus berupa bilangan bulat.

## Contoh

```sh
cat dataset.jsonl | jsonl-split-id \
  --output-dir hasil-split \
  --train 0.8 --validation 0.1 --test 0.1 \
  --seed 42 --stratify-by label
```

Input harus berisi satu objek JSON per baris. Tanpa `--stratify-by`, semua record diacak reproducibly. Dengan opsi itu, tiap nilai label scalar dibagi per kelas. File yang sudah ada tidak ditimpa kecuali diberikan `--force`. Ringkasan jumlah record per split dicetak ke stdout; record ditulis hanya ke file output.

## Sebagai library

```python
from jsonl_split_id import load_jsonl, split_records

rows = load_jsonl(jsonl_text)
splits = split_records(rows, seed=42, stratify_by="label")
```

Python 3.9+. Instalasi: `python -m pip install .`

## Batasan dan privasi

Alokasi rasio memakai largest-remainder rounding, sehingga ukuran total tepat dan tie-break deterministik. Kelas yang sangat kecil bisa tidak muncul di semua split; periksa ringkasan dan komposisi sesuai kebutuhan. Tool tidak membuang duplikat atau memvalidasi kualitas label. Input diproses lokal; isi dataset tidak dicetak ke stdout/error, tetapi file split berisi record sumber yang sengaja ditulis ke direktori pilihan.

Tes: `python -m unittest discover -s tests -v`

## Lisensi

MIT. Lihat [LICENSE](LICENSE).
