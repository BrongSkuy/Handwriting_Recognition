# Sistem Pengenalan Tulisan Tangan A-Z (Handwriting Recognition)

## 📋 Deskripsi Proyek

Sistem Pengenalan Tulisan Tangan A-Z adalah aplikasi web yang menggunakan **Convolutional Neural Network (CNN)** untuk mengenali dan mengklasifikasikan huruf tulisan tangan dari A sampai Z. Proyek ini merupakan tugas akhir (UAS) mata kuliah **Pengolahan Citra Digital** di UNPAM Semester 5.

Aplikasi ini memungkinkan pengguna untuk:
- Menggambar huruf secara manual di canvas
- Mengunggah gambar berformat image
- Mendapatkan prediksi huruf deangan tingkat akurasi/confidence
- Memvisualisasikan proses prediksi

## 🎯 Fitur Utama

- **Interface Interaktif**: Canvas untuk menggambar huruf secara real-time
- **Upload Gambar**: Mendukung format JPG, PNG, dan format image lainnya
- **Prediksi Real-Time**: Instant recognition dengan confidence score
- **Model CNN**: Terlatih menggunakan dataset A_Z Handwritten Data
- **Responsive Design**: Kompatibel dengan desktop dan mobile
- **Visualisasi Hasil**: Menampilkan probabilitas untuk setiap klasifikasi

## 🛠️ Teknologi yang Digunakan

### Backend
- **Python 3.x**
- **Flask**: Web framework
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Image processing
- **NumPy & Pandas**: Data manipulation

### Frontend
- **HTML5**: Structure
- **CSS3**: Styling & animations
- **JavaScript**: Interaktivitas & canvas drawing

### Machine Learning
- **Convolutional Neural Network (CNN)**
- **Dataset**: A_Z Handwritten Data (Dataset MNIST-style untuk huruf A-Z)

## 📦 Instalasi & Setup

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Langkah-langkah Instalasi

1. **Clone atau download project**
   ```bash
   cd handwriting_recognition
   ```

2. **Buat virtual environment (opsional tapi recommended)**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset dan train model (jika belum ada model yang terlatih)**
   ```bash
   # Jalankan Jupyter Notebook untuk training
   jupyter notebook model_training.ipynb
   ```
   Atau langsung jalankan di Jupyter:
   - Buka `model_training.ipynb`
   - Run semua cells untuk melatih model dan generate `model.h5`

## 🚀 Menjalankan Aplikasi

```bash
python app.py
```

Aplikasi akan berjalan di:
```
http://localhost:5000
```

Buka browser dan akses aplikasi melalui URL tersebut.

## 📁 Struktur Project

```
handwriting_recognition/
├── app.py                          # Main Flask application
├── model_training.ipynb            # Notebook untuk training model CNN
├── requirements.txt                # Dependencies
├── A_Z Handwritten Data.csv        # Dataset untuk training
├── model/
│   ├── model.h5                    # Trained CNN model
│   ├── labels.pkl                  # Label/kelas (A-Z)
│   └── preprocess_info.pkl         # Info preprocessing
├── static/
│   ├── css/
│   │   └── style.css               # Styling aplikasi
│   ├── js/
│   │   └── script.js               # JavaScript untuk canvas & API calls
│   └── uploads/                    # Folder untuk temporary uploads
└── templates/
    └── index.html                  # HTML template halaman utama
└── training_output/
    └── best_model.h5               # Best model dari training
```

## 🤖 Arsitektur Model CNN

Model yang digunakan memiliki struktur:
- **Input Layer**: 28x28 grayscale images
- **Convolutional Layers**: Ekstraksi features
- **Pooling Layers**: Dimensionality reduction
- **Dense Layers**: Classification
- **Output Layer**: 26 units (softmax activation untuk A-Z)

Proses preprocessing pada setiap input:
- Konversi ke grayscale
- Resize ke 28x28 pixels
- Invert colors (background putih, tulisan hitam)
- Normalisasi intensitas pixel

## 💡 Cara Penggunaan

### Metode 1: Menggambar Tulisan Tangan
1. Buka aplikasi di browser
2. Di canvas bagian kiri, gambar huruf A-Z dengan mouse atau stylus
3. Klik tombol **"Prediksi"**
4. Hasil prediksi akan ditampilkan beserta confidence score

### Metode 2: Upload Gambar
1. Klik tombol **"Upload Gambar"** atau gunakan file picker
2. Pilih gambar huruf yang ingin dikenali (format: JPG, PNG, BMP, dll)
3. Sistem akan otomatis melakukan prediksi
4. Lihat hasil dan confidence score pada panel kanan

### Kontrol Canvas
- **Brush Size**: Slider untuk mengatur ukuran kuas
- **Clear**: Hapus semua gambar di canvas
- **Download**: Simpan gambar yang telah digambar

## 📊 Dataset

Dataset yang digunakan adalah **A_Z Handwritten Data**, terdiri dari:
- **26 kelas**: Huruf A sampai Z
- **Images**: Tersedia dalam file CSV dengan pixel-pixel data
- **Ukuran**: 28x28 pixels untuk setiap gambar
- **Format**: Grayscale (nilai 0-255)

## 🔧 Konfigurasi

Beberapa parameter yang dapat dikonfigurasi di `app.py`:

```python
app.config['UPLOAD_FOLDER'] = 'static/uploads/'          # Folder uploads
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024      # Max file size 16MB
MODEL_PATH = 'model/model.h5'                            # Path ke trained model
IMG_SIZE = 28                                            # Ukuran input image
```

## 🐛 Troubleshooting

### Error: "Model tidak ditemukan"
**Solusi**: Jalankan `model_training.ipynb` terlebih dahulu untuk melatih model dan generate `model.h5`

### Error: "Module tidak ditemukan"
**Solusi**: Pastikan semua dependencies terinstall dengan:
```bash
pip install -r requirements.txt
```

### Prediksi tidak akurat
**Solusi**:
- Pastikan gambar huruf jelas dan terpisah
- Gunakan ukuran kuas yang sesuai (15-20 pixels recommended)
- Gambar dengan style yang mirip dataset training

### Port 5000 sudah digunakan
**Solusi**: Ubah port di `app.py`:
```python
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Ganti ke port lain
```

## 📈 Performa Model

Performa model dapat dilihat pada hasil training di `model_training.ipynb`:
- Training accuracy: ~97-99%
- Validation accuracy: ~95-98%
- Test accuracy: ~93-97%

## 🎓 Pembelajaran

Project ini meliputi konsep-konsep penting:
- **Convolutional Neural Networks (CNN)**: Arsitektur deep learning untuk image classification
- **Image Processing**: Preprocessing, normalisasi, dan augmentasi gambar
- **Machine Learning Pipeline**: Training, validation, dan testing
- **Web Development**: Flask, REST API, frontend-backend integration

## 📝 Lisensi

Proyek ini dibuat untuk keperluan akademik sebagai UAS Pengolahan Citra Digital.

## 👨‍💼 Informasi Proyek

- **Institusi**: UNPAM (Universitas Pamulang)
- **Semester**: 5 (Lima)
- **Mata Kuliah**: Pengolahan Citra Digital
- **Jenis Tugas**: UAS (Ujian Akhir Semester)
- **Model**: Convolutional Neural Network (CNN)

## 📞 Support & Feedback

Untuk pertanyaan atau saran, silakan hubungi pembuat proyek atau asisten dosen mata kuliah Pengolahan Citra Digital.

---

**Selamat menggunakan Sistem Pengenalan Tulisan Tangan A-Z!** ✨
