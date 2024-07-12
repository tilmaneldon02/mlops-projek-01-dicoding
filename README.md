# Submission 1: Machine Learning Pipeline - Apple Quality Predictor
Nama: Tilman Eldon

Username Dicoding: eldonn

|  | Deskripsi |
| --- | --- |
| Dataset | [Apple Quality Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality/data) |
| Masalah | Tujuan dari proyek ini adalah mengembangkan model prediksi untuk menentukan kualitas apel berdasarkan sejumlah variabel seperti ukuran, warna, berat, kekerasan, dan tingkat kematangan. Tantangan utama meliputi pengumpulan data yang akurat dan representatif, pemilihan fitur yang relevan, serta pemilihan dan pengujian algoritma yang dapat menghasilkan prediksi dengan akurasi tinggi. Model yang dihasilkan harus mampu membantu produsen apel dalam memastikan konsistensi kualitas produk dan mengurangi risiko penolakan oleh konsumen. |
| Solusi machine learning | Model machine learning yang bertujuan memprediksi kualitas apel berdasarkan dataset yang tersedia. Solusi ini meliputi preprocessing data, pelatihan model, serta evaluasi model untuk menjamin performa yang optimal. |
| Metode pengolahan | Splitting Data: Membagi dataset menjadi data train-eval dengan ratio 8:2. |
| Arsitektur model | Model neural network dengan arsitektur sebagai berikut: <ul><li>Input Layer: Menangani 7 fitur atribut dari karakteristik apel.<li>Dense Layer 1: 64 unit dengan aktivasi ReLU.<li>Dense Layer 2: 32 unit dengan aktivasi ReLU.<li>Output Layer: 1 unit untuk memprediksi kualitas apel.<ul> |
| Metrik evaluasi | <ul><li>ExampleCount: Menghitung jumlah contoh dalam dataset.<li>AUC (Area Under the Curve): Mengukur kemampuan model untuk membedakan antara kelas.<li>FalsePositives: Menghitung jumlah positif palsu, yaitu prediksi positif yang salah.<li>TruePositives: Menghitung jumlah positif benar, yaitu prediksi positif yang benar.<li>FalseNegatives: Menghitung jumlah negatif palsu, yaitu prediksi negatif yang salah.<li>TrueNegatives: Menghitung jumlah negatif benar, yaitu prediksi negatif yang benar.<li>BinaryAccuracy: Mengukur akurasi biner dari model, dengan ambang batas (threshold) yang ditentukan. Ambang batas ini memastikan bahwa nilai akurasi minimal 0.5, dan setiap perubahan akurasi harus lebih besar dari 0.0001 agar dianggap signifikan. |
| Performa model | Model yang dibuat menghasilkan performa yang cukup baik dalam memberikan prediksi untuk data karakteristik apel yang diinputkan, dan dari pelatihan yang dilakukan model menghasilkan accuracy sekitar 99% dan val_accuracy sekitar 93% |
