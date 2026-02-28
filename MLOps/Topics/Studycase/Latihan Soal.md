# Soal Latihan Praktik MLOps - Deploy ML Model dengan Flask

**Level:** Pemula - Menengah | **Waktu Estimasi:** 60 menit

---

## 📋 Latar Belakang

Anda adalah seorang Data Engineer yang ditugaskan untuk membuat sistem prediksi income menggunakan Machine Learning. Anda diminta untuk membuat pipeline MLOps lengkap mulai dari preprocessing data, training model, hingga deployment menggunakan Flask.

---

## 🎯 Tujuan Latihan

Setelah menyelesaikan latihan ini, Anda akan mampu:
1. Melakukan preprocessing data dengan handling missing values
2. Encoding categorical variables
3. Training Decision Tree Classifier
4. Menyimpan model dalam format pickle
5. Membuat Flask API untuk serving model
6. Membuat form web untuk input pengguna

---

## 📚 Soal-Soal Latihan

### **Soal 1: Data Preprocessing (15 menit)**

**Permasalahan:**
Anda memiliki dataset `adult.csv` dengan beberapa masalah data:
- Ada missing values yang ditandai dengan "?"
- Kategori marital-status memiliki banyak nilai unik yang bisa disederhanakan
- Beberapa kolom categorical belum di-encode

**Tugas Anda:**
1. Buat file `preprocessing.py`
2. Load dataset dari file `adult.csv`
3. Replace "?" dengan NaN
4. Fill missing values menggunakan mode (modus)
5. Sederhanakan marital-status menjadi 3 kategori: "married", "not married", dan "divorced"
6. Lakukan Label Encoding pada kolom categorical: `workclass`, `education`, `marital-status`, `occupation`, `gender`
7. Drop kolom yang tidak diperlukan: `fnlwgt`, `educational-num`
8. Simpan dataset yang sudah di-preprocess

**Tips:**
- Gunakan `pandas` untuk data manipulation
- Gunakan `sklearn.preprocessing.LabelEncoder` untuk encoding
- Buat mapping dictionary untuk menyimpan encoding mapping

---

### **Soal 2: Model Training & Persistence (15 menit)**

**Permasalahan:**
Dataset sudah di-preprocess, sekarang Anda perlu melatih model Decision Tree dan menyimpannya untuk digunakan di production.

**Tugas Anda:**
1. Lakukan train-test split dengan rasio 70:30
2. Inisialisasi Decision Tree Classifier dengan parameter:
   - `criterion="gini"`
   - `max_depth=5`
   - `min_samples_leaf=5`
   - `random_state=100`
3. Fit model dengan training data
4. Evaluasi model dengan mencalculate accuracy score pada test set
5. Simpan model menggunakan pickle dengan nama file `model.pkl`
6. Print accuracy score dan jumlah features yang digunakan

**Tips:**
- Gunakan `train_test_split` dari sklearn
- Gunakan `pickle` untuk menyimpan model
- Cek bahwa file `model.pkl` berhasil dibuat

---

### **Soal 3: Flask API Development (20 menit)**

**Permasalahan:**
Model sudah siap, sekarang Anda perlu membuat API untuk serving model menggunakan Flask.

**Tugas Anda:**
1. Buat file `app.py` dengan struktur Flask app yang benar
2. Buat route `/` yang menampilkan halaman form (`index.html`)
3. Buat function `ValuePredictor()` yang:
   - Menerima list of values dari form
   - Reshape data menjadi format yang sesuai (1, 12)
   - Load model dari `model.pkl`
   - Return prediction result
4. Buat route `/result` yang:
   - Menerima POST request dari form
   - Extract nilai dari form
   - Panggil `ValuePredictor()`
   - Render hasil prediction ke `result.html`
5. Jalankan Flask app dan test apakah berjalan tanpa error

**Tips:**
- Pastikan path menuju model.pkl benar
- Gunakan `request.form.to_dict()` untuk extract data dari form
- Convert string values menjadi integer sebelum predict

---

### **Soal 4: Frontend Development (10 menit)**

**Permasalahan:**
API sudah ready, sekarang Anda perlu membuat form HTML untuk input dari pengguna.

**Tugas Anda:**
1. Buat folder `templates`
2. Buat file `index.html` dengan form yang memiliki field untuk 12 features:
   - Age (input number)
   - Workclass (select dropdown)
   - Education (select dropdown)
   - Marital Status (select dropdown)
   - Occupation (select dropdown)
   - Gender (select radio button)
   - Dan field lainnya sesuai dataset
3. Form harus POST ke `/result`
4. Buat file `result.html` yang menampilkan hasil prediksi
5. Gunakan styling HTML sederhana untuk membuat tampilan lebih rapi

**Tips:**
- Gunakan form input dengan name attribute yang sesuai
- Dropdown value harus berupa angka (encoded value)
- Tampilkan hasil prediksi dengan format yang user-friendly

---

## 🔍 Kriteria Penilaian

| Aspek | Skor |
|-------|------|
| Preprocessing data benar dan lengkap | 20% |
| Model training dan persistence | 20% |
| Flask API implementation | 30% |
| Frontend dan form submission | 20% |
| Dokumentasi dan komentar kode | 10% |

---

## 📦 Deliverables

Setelah menyelesaikan semua soal, siapkan:
1. ✅ `preprocessing.py` - File preprocessing yang lengkap
2. ✅ `train_model.py` - File training dan saving model (Soal 2)
3. ✅ `model.pkl` - Model yang sudah di-save
4. ✅ `app.py` - Flask application
5. ✅ `templates/index.html` - Form input
6. ✅ `templates/result.html` - Hasil prediksi
7. ✅ `requirements.txt` - List dependencies

---

## 🚀 Bonus Challenge (Optional)

Jika Anda sudah menyelesaikan semua soal, coba kerjakan bonus challenge:

1. **Error Handling**: Tambahkan error handling di Flask app untuk validasi input
2. **Model Metrics**: Tambahkan precision, recall, dan F1-score selain accuracy
3. **Data Validation**: Validasi bahwa input dari user sesuai dengan range training data
4. **REST API**: Ubah form submission menjadi REST API endpoint yang return JSON
5. **Model Versioning**: Simpan model dengan timestamp untuk tracking versi model

---

## 📝 Catatan Penting

- **File Dataset**: Download dari [Adult Income Dataset](https://media.geeksforgeeks.org/wp-content/uploads/20250324175201901232/adult.csv)
- **Dependencies**: Flask, Pandas, NumPy, Scikit-learn, Pickle
- **Testing**: Test setiap step secara incremental, jangan menyelesaikan semua sekaligus
- **Debugging**: Jika ada error, cek path file dan nama column yang benar

---

## 💡 Helpful Resources

- [Sklearn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Flask Basics](https://flask.palletsprojects.com/en/3.0.x/)
- [Pickle Documentation](https://docs.python.org/3/library/pickle.html)

---

**Good Luck! 🎓**
