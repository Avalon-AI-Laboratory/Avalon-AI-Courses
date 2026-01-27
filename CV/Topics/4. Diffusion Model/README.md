# Masih kasaran 

# Modul Pembelajaran: Teori di Balik Diffusion Model

## 1. Intuisi Dasar: Merusak dan Memperbaiki

Bayangkan kita memiliki sebuah lukisan yang sangat jernih (ini adalah **data asli** kita, atau **x₀**).

Sekarang, kita secara bertahap meneteskan tinta ke atas lukisan tersebut. Setiap tetes tinta membuat lukisan sedikit lebih kabur dan acak. Jika kita melakukannya ratusan kali, lukisan asli tersebut pada akhirnya akan benar-benar tertutup oleh tinta dan terlihat seperti noda acak (**noise murni**).

Proses merusak ini kita sebut **Forward Process (Proses Maju)**. Proses ini mudah dilakukan dan tidak memerlukan "kecerdasan" apa pun.

Bagaimana jika kita ingin membalik prosesnya? Bisakah kita, mulai dari noda acak, secara bertahap "mengangkat" setiap tetes tinta hingga lukisan aslinya kembali muncul? Ini jauh lebih sulit. Kita perlu tahu persis di mana setiap tetes tinta berada dan bagaimana cara menghilangkannya tanpa merusak gambar di bawahnya.

Inilah tugas utama dari sebuah Diffusion Model. Kita melatih sebuah model AI (Neural Network) untuk menjadi "pembersih" ahli. Model ini belajar cara melakukan **Reverse Process (Proses Mundur)**: mengubah noise acak, langkah demi langkah, kembali menjadi gambar yang koheren dan realistis.

## 2. Teori dan Matematika

Mari kita formalisasikan intuisi di atas.

### a. Forward Process (q)

Proses maju (disebut juga proses difusi) didefinisikan sebagai penambahan sejumlah kecil noise Gaussian ke gambar pada setiap langkah waktu (*timestep*) `t`. Kita punya `T` total langkah (misalnya, `T=1000`).

Pada setiap langkah `t`, gambar `x_t` dihasilkan dari `x_{t-1}` dengan rumus:

**x_t = sqrt(α_t) * x_{t-1} + sqrt(1 - α_t) * ε**

- **x_{t-1}**: Gambar dari langkah sebelumnya.
- **ε (epsilon)**: Noise acak dari distribusi Gaussian stkitar.
- **α_t (alpha)**: Sebuah parameter kecil yang mengontrol seberapa banyak noise yang ditambahkan. Nilai `α_t` ini ditentukan oleh sebuah jadwal (*schedule*), biasanya `α_t = 1 - β_t`, di mana `β_t` (beta) adalah varians noise yang kecil dan meningkat secara bertahap dari `t=1` hingga `t=T`.

Kelebihan dari proses ini adalah kita tidak perlu melakukannya langkah per langkah. Kita bisa langsung melompat ke timestep `t` manapun dari gambar asli `x₀` menggunakan rumus:

**x_t = sqrt(ᾱ_t) * x₀ + sqrt(1 - ᾱ_t) * ε**

- **ᾱ_t (alpha-bar)**: Merupakan produk kumulatif dari semua `α` hingga `t` (yaitu, `α₁ * α₂ * ... * α_t`).

Rumus ini sangat penting untuk efisiensi saat training.

### b. Reverse Process (p_θ)

Tujuan kita adalah membalik proses di atas: dari `x_T` (noise murni), kita ingin menghasilkan `x_{T-1}`, lalu `x_{T-2}`, dan seterusnya hingga kembali ke `x₀` (gambar jernih).

Secara matematis, kita ingin menghitung probabilitas `p(x_{t-1} | x_t)`. Namun, ini sangat sulit dihitung secara langsung. Di sinilah *deep learning* berperan. Kita melatih sebuah model neural network (dengan parameter `θ`, sehingga kita sebut `p_θ`) untuk **mengaproksimasi** proses kebalikan ini.

**Trik Penting**: Daripada melatih model untuk memprediksi piksel-piksel dari gambar `x_{t-1}` secara langsung (yang cenderung tidak stabil), penelitian menemukan bahwa jauh lebih efektif untuk melatih model **memprediksi noise (ε)** yang ditambahkan pada langkah tersebut.

Jadi, tugas model kita (`unet.py` dalam kode) adalah:

> Diberikan gambar `x_t` yang berisi noise pada timestep `t`, prediksikan `ε_θ(x_t, t)`, yaitu noise yang ditambahkan ke dalamnya.

### c. Loss Function: Belajar Memprediksi Noise

Bagaimana model tahu apakah prediksinya benar? Kita menggunakan *loss function* yang sangat sederhana: **Mean Squared Error (MSE)**.

Saat training:
1.  Ambil gambar asli `x₀`.
2.  Pilih timestep acak `t`.
3.  Buat noise acak `ε`.
4.  Gunakan rumus *forward process* untuk menghasilkan gambar `x_t` yang ber-noise.
5.  Berikan `x_t` dan `t` ke model kita untuk mendapatkan prediksi noise: `ε_θ(x_t, t)`.
6.  Hitung *loss*: `MSE(ε, ε_θ(x_t, t))`. Ini adalah selisih kuadrat antara noise asli yang kita tambahkan (`ε`) dan noise yang diprediksi oleh model (`ε_θ`).

Dengan meminimalkan *loss* ini, model secara bertahap menjadi sangat pkitai dalam mengenali dan "melihat" noise dalam sebuah gambar pada timestep manapun.

## 3. Arsitektur Model: Mengapa U-Net?

Model yang kita gunakan untuk memprediksi noise adalah **U-Net**. Arsitektur ini sangat cocok karena beberapa alasan:

1.  **Input dan Output Berukuran Sama**: U-Net menerima gambar dan menghasilkan output (dalam kasus kita, prediksi noise) dengan dimensi yang persis sama. Ini krusial karena kita perlu peta noise yang sesuai dengan gambar.

2.  **Multi-Skala Kontekstual**: U-Net memiliki jalur *downsampling* (encoder) dan *upsampling* (decoder).
    -   **Encoder**: Secara bertahap mengurangi ukuran spasial gambar, memungkinkan model untuk "melihat" fitur-fitur global dan konteks (misalnya, "ini sepertinya wajah manusia").
    -   **Decoder**: Secara bertahap membangun kembali ukuran gambar, fokus pada detail yang lebih halus (misalnya, "ini adalah mata, ini adalah hidung").

3.  **Skip Connections**: Ini adalah fitur paling penting dari U-Net. Ada "jembatan" yang menghubungkan layer-layer dari encoder ke layer-layer yang sesuai di decoder. Jembatan ini memungkinkan informasi detail dari tahap awal (resolusi tinggi) untuk langsung mengalir ke tahap akhir, mencegah hilangnya informasi spasial yang penting selama proses kompresi di *bottleneck*.

4.  **Injeksi Timestep**: Informasi `t` (timestep) sangat penting. Kita mengubah `t` menjadi sebuah *embedding vector* (mirip dengan *positional encoding* pada Transformer) dan menginjeksikannya ke dalam berbagai layer di U-Net. Ini memberi tahu model seberapa "berisik" gambar yang sedang dilihatnya, sehingga ia bisa menyesuaikan prediksinya.

## 4. Cara Kerja Praktis (Training & Sampling)

-   **Training (`train.py`)**: Seperti yang dijelaskan di atas, skrip ini berulang kali mengambil gambar, menambahkan noise, meminta model menebak noise-nya, dan memperbarui model berdasarkan kesalahannya.

-   **Sampling (Generasi Gambar)**: Setelah model dilatih, proses generasi dimulai:
    1.  Buat sebuah gambar dari noise Gaussian murni (ini adalah `x_T`).
    2.  Mulai dari `t = T` hingga `t = 1`:
        a. Berikan `x_t` dan `t` ke model untuk memprediksi noise `ε_θ`.
        b. Gunakan prediksi noise ini dalam rumus matematika *reverse process* untuk menghitung `x_{t-1}` yang sedikit lebih bersih.
    3.  Setelah loop selesai, `x₀` yang dihasilkan adalah gambar baru kita.

## 5. Kesimpulan

Diffusion Model adalah perpaduan elegan antara proses stokastik yang terdefinisi dengan baik (Forward Process) dan kekuatan aproksimasi dari neural network (Reverse Process). Dengan melatih model untuk melakukan tugas yang tampaknya sederhana—memprediksi noise—kita secara implisit mengajarinya struktur, tekstur, dan fitur dari data training, yang memungkinkannya untuk menghasilkan data baru yang realistis dari ketiadaan.

---

*Untuk panduan praktis tentang cara menjalankan kode, silakan lihat bagian [Cara Menjalankan](#5-cara-menjalankan) dan [Struktur Proyek](#3-struktur-proyek) pada versi sebelumnya dari dokumen ini, atau jalankan `python train.py`.*
