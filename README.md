# Banana Leaf Disease Classification  
**Perbandingan MobileNetV2 dan EfficientNetB0 pada Klasifikasi Penyakit Daun Pisang**

## 📌 Deskripsi
Repository ini berisi implementasi penelitian skripsi yang membahas **perbandingan performa MobileNetV2 dan EfficientNetB0** dalam melakukan klasifikasi penyakit daun pisang. Penelitian difokuskan pada tiga aspek utama, yaitu **efektivitas**, **efisiensi**, dan **robustness** model terhadap gangguan visual.

Penyakit daun pisang yang diklasifikasikan meliputi:
- **Cordana**
- **Pestalotiopsis**
- **Sigatoka**
- **Healthy (daun sehat)**

Penelitian ini menggunakan pendekatan **transfer learning** dengan dua skenario utama:
1. **Fixed Feature**
2. **Fine-Tuning (FT10, FT20, FT30)**

---

## 🎯 Tujuan Penelitian
1. Membandingkan performa MobileNetV2 dan EfficientNetB0 dari sisi **akurasi dan metrik evaluasi**.
2. Menganalisis **efisiensi komputasi** model berdasarkan waktu pelatihan, ukuran model, FLOPs, dan MACs.
3. Menguji **robustness model** terhadap gangguan visual berupa perubahan brightness, blur, dan noise salt and pepper.
4. Mengevaluasi kemampuan generalisasi model menggunakan **dataset luar** dengan kondisi lapangan yang lebih natural.

---
