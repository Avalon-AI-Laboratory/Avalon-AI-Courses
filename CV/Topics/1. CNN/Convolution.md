# Convolutional Neural Network (CNN)
CNN as it names "neural network" work like how human visual cortex systems works. and it used in deeplearning realm.  

jika di sebelumnya kita telah mengenal konsep image gradient vector, dan bagaimana algoritma HOG menyimpulkan informasi dari seluruh gradient vector dari sebuah gambar dll

disini kita membahas cnn classic untuk klasifikasi gambar. yang juga merupakan fondasi untuk nantinya ya 

well, kita akan membahas dulu keterbatasan transformasi afin (affine transformations) dalam jaringan saraf tradisional

Inti dari jaringan saraf adalah transformasi afine: sebuah vektor diterima sebagai input dan dikalikan dengan sebuah matriks untuk menghasilkan output (biasanya ditambahkan vektor bias sebelum hasilnya dilewatkan melalui fungsi nonlinier). Hal ini berlaku untuk segala jenis input, baik itu gambar, klip suara, maupun kumpulan fitur yang tidak terurut: apapun dimensinya, representasinya selalu bisa diratakan menjadi vektor sebelum dilakukan transformasi:

## Mekanisme
<ol>
<li>Input → Vektor Rata: Input apa pun (gambar, klip suara, fitur) harus diratakan (flattened) menjadi satu vektor (satu array 1-dimensi).

<li>Perkalian Matriks: Vektor input ini dikali dengan sebuah matriks (disebut weight matrix).

<li>Penambahan Bias: Hasil perkalian matriks ini ditambah dengan sebuah vektor bias.

<li>Nonlinieritas: Hasil akhir kemudian dilewatkan melalui fungsi nonlinier (seperti ReLU atau Sigmoid).
</ol>

Secara matematis, untuk input vektor x, matriks bobot W, dan vektor bias b, transformasinya adalah:

$$Output=f(Wx+b)$$

(di mana f adalah fungsi nonlinier).

## Masalah utamanya
Masalah-nya dengan Transformasi Afin ini adalah Mengabaikan Struktur Data. karena transformasi afin tidak memanfaatkan struktur intrinsik (bawaan) data terstruktur seperti gambar

| Properti           | Penjelasan                                                                 | Contoh                                                      |
|-------------------|---------------------------------------------------------------------------|------------------------------------------------------------|
| Array Multi-dimensi | Data disimpan dalam bentuk tensor dengan lebih dari satu dimensi.         | Gambar (Tinggi × Lebar × Channel)                         |
| Urutan Penting      | Urutan di sepanjang sumbu tertentu memiliki arti struktural.             | Sumbu lebar dan tinggi pada gambar; sumbu waktu pada klip suara. Mengubah urutan piksel secara acak akan merusak gambar. |
| Sumbu Channel       | Sumbu khusus untuk mengakses "pandangan" berbeda dari data yang sama.     | RGB (Merah, Hijau, Biru) pada gambar berwarna; Kiri/Kanan pada audio stereo. |

Intinya keterbatasannya adalah Ketika data terstruktur (misalnya, gambar 3D) diratakan menjadi vektor 1D. 

**Informasi Topologi Hilang** Hubungan spasial (kedekatan) antara piksel yang berdekatan dalam gambar rusak

**Diperlakukan Sama** dimana semua sumbu (lebar, tinggi, channel) diperlakukan sama setelah diratakan. Jaringan saraf tidak tahu bahwa dua elemen yang berdekatan di vektor mungkin sebenarnya jauh di gambar asli, atau sebaliknya.

ya pada akhirnya **Struktur Tidak Dieksploitasi** Sifat penting seperti urutan dan channel diabaikan.

## Solusi
Sebagai solusinya **discrete convolution** berperan untuk mempertahankan struktur intrinsik data (dimensi spasial dan urutan) dan memanfaatkannya saat memproses data.

## 1.1 Discrete convolutions

Cara penerapan dari Konvolusi diskrit adalah sebagai blok bangunan utama dalam Convolutional Neural Networks (CNNs), yang sangat efektif dalam tugas-tugas yang melibatkan data terstruktur seperti visi komputer dan pengenalan ucapan.

Akhirnya, Dengan menggunakan konvolusi, jaringan dapat belajar pola lokal dan fitur spasial (misalnya, garis tepi atau sudut) dengan menerapkan kernel (filter) kecil yang bergerak melintasi dimensi spasial, sehingga menghormati topologi data.

## sifat utama konvdisk
Konvolusi diskrit adalah jenis transformasi linier yang dirancang untuk melestarikan konsep urutan (atau topologi) data. dengan dua sifat mendasar yang membuatnya sangat efisien dan efektif

### Sparsitas (Konektivitas Lokal)
Berbeda dengan transformasi afine di mana setiap unit input terhubung ke setiap unit output (*fully connected*), konvolusi bersifat *sparse connected*. Artinya, hanya sebagian kecil unit input yang berdekatan yang berkontribusi pada perhitungan satu unit output.

Jika divisualisasikan, setiap piksel output hanya dipengaruhi oleh sekelompok piksel lokal di sekitar posisi yang sama pada gambar input. Hal ini memanfaatkan fakta bahwa informasi yang paling relevan untuk suatu piksel biasanya ada di sekitarnya.

### Parameter Sharing
Konvolusi menggunakan kembali bobot yang sama (disebut *kernel* atau *filter*) di berbagai lokasi input yang berbeda. Bayangkan seperti menggunakan kaca pembesar yang sama untuk mencari pola tertentu (misalnya garis tepi horizontal) di setiap area gambar.

Sifat ini sangat mengurangi jumlah parameter yang harus dipelajari oleh model, sehingga jauh lebih efisien dibandingkan lapisan afine penuh, yang harus mempelajari bobot unik untuk setiap koneksi.

![kernel area](../Asset/Screenshot%202025-10-15%20174032.png)

wujud asli dari kernel, matriks bobot kecil yang disebut filter. Kernel inilah yang digeser (slide) ke seluruh peta fitur input untuk melakukan operasi perkalian dan penjumlahan, menghasilkan satu titik pada peta fitur output.
![discrete conv](../Asset/discreteConv.png)

Kernel (atau filter) bergerak melintasi peta fitur input (input feature map). Pada setiap posisi, dilakukan perhitungan hasil kali antara setiap elemen kernel dengan elemen input yang tumpang tindih, kemudian hasil-hasil tersebut dijumlahkan untuk menghasilkan output pada posisi saat ini.
Prosedur ini dapat diulangi dengan menggunakan berbagai kernel yang berbeda untuk membentuk sebanyak mungkin peta fitur output (output feature maps) yang diinginkan (lihat Gambar 1.3).
Hasil akhir dari prosedur ini disebut output feature maps.

Jika terdapat beberapa peta fitur input, maka kernel harus memiliki dimensi tiga (3D) — atau secara ekuivalen, setiap peta fitur input dikonvolusikan dengan kernel yang berbeda — dan hasil dari tiap konvolusi tersebut dijumlahkan secara elemen demi elemen (elementwise) untuk menghasilkan satu peta fitur output.

Konvolusi yang digambarkan pada Gambar sebelumnya adalah contoh dari konvolusi dua dimensi (2D convolution), namun konsepnya dapat digeneralisasi menjadi konvolusi N-dimensi (N-D convolution).
Sebagai contoh, dalam konvolusi 3D, kernel berbentuk kubus (cuboid) dan bergerak melintasi tinggi (height), lebar (width), serta kedalaman (depth) dari peta fitur input.

Kumpulan kernel yang mendefinisikan konvolusi diskrit memiliki bentuk (shape) yang sesuai dengan suatu permutasi dari $(n, m, k_1, \ldots, k_N)$, di mana:


1. $n$ = jumlah *output feature maps*

Setiap *layer konvolusi* menghasilkan sejumlah *feature map* baru — sesuai dengan berapa banyak *kernel/filter* yang kita gunakan.

Misalnya, kita menggunakan **32 filter**: maka layer tersebut menghasilkan **32 output feature maps**, jadi $n = 32$


2. $m$ = jumlah *input feature maps*  

Bayangkan kita memasukkan gambar RGB ke dalam CNN.  
Gambar RGB memiliki 3 *channel*: **R** (Red), **G** (Green), **B** (Blue)

Artinya jumlah *input feature map* adalah 3 (karena ada 3 *channel*). Jadi $m = 3$

Jika lapisan sebelumnya menghasilkan 16 *feature maps*, maka: $m = 16$


3. $k_j$ = ukuran kernel sepanjang sumbu ke-$j$

$o_j$ yang digunakan untuk menentukan ukuran tiap feature map (tinggi × lebar). Dimana beberapa properti berikut memengaruhi ukuran output $o_j$ dari suatu lapisan konvolusi di sepanjang sumbu ke-$j$:

$$
o_j = \left\lfloor \frac{i_j + 2p_j - k_j}{s_j} \right\rfloor + 1
$$

- $o_j$ : ukuran *output* sepanjang sumbu ke-$j$ 
- $i_j$ : ukuran *input* sepanjang sumbu ke-$j$  
- $k_j$ : ukuran *kernel* sepanjang sumbu ke-$j$  
- $s_j$ : *stride* (jarak antara dua posisi kernel yang berurutan) di sepanjang sumbu ke-$j$  
- $p_j$ : *zero padding* (jumlah nol yang ditambahkan di awal dan akhir suatu sumbu) di sepanjang sumbu ke-$j$

Sebagai contoh, Gambar 1.2 menunjukkan sebuah kernel 3 × 3 yang diterapkan pada input 5 × 5 dengan padding nol 1 × 1 dan stride 2 × 2.

![OutputVal](../Asset/OutputValues.png)

Perlu diperhatikan bahwa stride merupakan salah satu bentuk subsampling.
Sebagai alternatif dari menganggapnya sebagai ukuran seberapa jauh kernel digeser, stride juga dapat dipandang sebagai ukuran seberapa banyak output yang dipertahankan.

gambar ini menunjukkan bagaimana satu set filter (kernel) bekerja pada peta fitur input, menghasilkan peta fitur output
<img src="../Asset/convolutionMapping.png" alt="convolution mapping from two input feature maps to three output" width="400">

Sebuah proses konvolusi memetakan dua *input feature maps* menjadi tiga *output feature maps* menggunakan kumpulan kernel $w$ berukuran $3 \times 2 \times 3 \times 3$. Pada jalur kiri, *input feature map* pertama dikonvolusikan dengan kernel $w_{1,1}$ dan *input feature map* kedua dengan kernel $w_{1,2}$. Hasil dari kedua konvolusi tersebut dijumlahkan secara elemen demi elemen (*elementwise*) untuk membentuk *output feature map* pertama. Prosedur yang sama diulang untuk jalur tengah dan jalur kanan guna membentuk *output feature map* kedua dan ketiga, yang kemudian dikelompokkan bersama (*stacked*) untuk menghasilkan output akhir.

Contohnya gini biar lebih jelas:
![discrete conv](../Asset/Contoh-kernel-bekerja.png)

Perlu dicatat bahwa stride merupakan bentuk dari subsampling.
Sebagai alternatif dari melihat stride sebagai ukuran seberapa jauh kernel bergeser, stride juga dapat dianggap sebagai ukuran seberapa banyak bagian output yang dipertahankan.

Sebagai contoh, menggeser kernel dengan lompatan dua (stride = 2) setara dengan menggeser kernel dengan lompatan satu (stride = 1) tetapi hanya mempertahankan elemen output bernomor ganjil (lihat Gambar 1.4).

![stride](../Asset/Stride.png)


### 1.2 Pooling
Selain konvolusi diskrit itu sendiri, operasi pooling merupakan komponen penting lainnya dalam CNN (Convolutional Neural Networks).
Pooling berfungsi untuk mengurangi ukuran peta fitur (feature maps) dengan menggunakan suatu fungsi untuk merangkum subregion (wilayah kecil), misalnya dengan mengambil nilai rata-rata (average pooling) atau nilai maksimum (max pooling).

Pooling bekerja dengan cara menggeser sebuah jendela (window) di atas input, lalu mengumpankan isi jendela tersebut ke fungsi pooling.  
Dalam beberapa hal, mekanisme pooling mirip dengan konvolusi diskrit, namun menggantikan kombinasi linear (dot product) yang dilakukan kernel dengan fungsi lain (seperti maksimum atau rata-rata).

Gambar berikutnya bakalan menunjukkan contoh average pooling, abis itu bakal menunjukkan max pooling.

Ini average pooling
![AvgPool](../Asset/AvgPool.png)

Ini Max pooling
![AvgPool](../Asset/MaxPool.png)

## 2. Aritmetika Konvolusi (Convolution Arithmetic)
Analisis hubungan antara berbagai properti lapisan konvolusi menjadi lebih mudah karena setiap sumbu (axis) bekerja secara independen — artinya, pemilihan ukuran kernel, stride, dan zero padding pada sumbu ke-$j$ hanya memengaruhi ukuran output pada sumbu tersebut, dan tidak berinteraksi dengan sumbu lain.

### 2.1 Tanpa Zero Padding, dengan Stride = 1
Kasus paling sederhana untuk dianalisis adalah ketika **kernel** melintasi setiap posisi input tanpa *padding* dan tanpa lompatan, yaitu ketika $s = 1$ dan $p = 0$. Gambar 2.1 menunjukkan contohnya untuk $i = 4$ dan $k = 3$. Salah satu cara untuk menentukan **ukuran output** dalam kasus ini adalah dengan menghitung jumlah posisi yang mungkin bagi kernel di atas input, misalnya pada sumbu lebar (*width*).

Kernel mulai dari sisi paling kiri input, lalu bergeser satu langkah demi satu langkah hingga menyentuh sisi kanan. Ukuran output sama dengan jumlah langkah yang dilakukan ditambah satu — karena posisi awal kernel juga dihitung (lihat Gambar 2.8a). Logika yang sama berlaku untuk sumbu tinggi (*height*).  

Secara formal, hubungan berikut dapat diturunkan:  

**Hubungan 1**  
Untuk sembarang $i$ dan $k$, serta $s = 1$ dan $p = 0$:  

$$
o = (i - k) + 1
$$

![contoh](../Asset/No-zero-padding,unit-strides.png)

### 2.2 Zero Padding, dengan Stride = 1

Untuk memperhitungkan *zero padding* (dengan tetap membatasi $s = 1$), mari kita lihat efeknya terhadap ukuran *input* efektif:  
Padding sebesar $p$ di setiap sisi akan mengubah ukuran input efektif dari $i$ menjadi $i + 2p$.

Dalam kasus umum, Hubungan 1 dapat digunakan kembali untuk menurunkan hubungan berikut:

**Hubungan 2**  
Untuk sembarang $i, k, p$, dan $s = 1$:

$$
o = (i - k) + 2p + 1
$$

![stride](../Asset/ZeroPadding,dengan-Stride1.png)
Gambar 2.2 menunjukkan contoh untuk $i = 5$, $k = 4$, dan $p = 2$.

Dalam praktiknya, terdapat dua jenis *zero padding* yang sering digunakan karena sifatnya yang berguna. Mari kita bahas keduanya.

---

#### 2.2.1 Half (Same) Padding

Kadang kita menginginkan agar ukuran *output* sama dengan ukuran *input* (yaitu $o = i$).  
Kondisi ini bisa dicapai dengan memilih *padding* yang sesuai.

**Hubungan 3**  
Untuk sembarang $i$, dan jika $k$ adalah bilangan ganjil ($k = 2n + 1$, $n \in \mathbb{N}$),  
dengan $s = 1$ dan $p = \lfloor k / 2 \rfloor = n$, maka:

$$
o = i + 2\lfloor k/2 \rfloor - (k - 1) = i + 2n - 2n = i
$$

Kondisi ini disebut **half padding** atau **same padding** (*padding setengah* atau *padding sama*).  
![stride](../Asset/Half(same)padding.png)
Gambar 2.3 menunjukkan contohnya untuk $i = 5$, $k = 3$, dan karenanya $p = 1$.

---

#### 2.2.2 Full Padding

Secara umum, konvolusi dengan kernel akan mengurangi ukuran *output* dibandingkan ukuran *input*.  
Namun, dalam beberapa kasus, justru diperlukan *output* yang lebih besar dari *input*.  
Hal ini dapat dicapai dengan menambahkan *zero padding* yang cukup besar.

**Hubungan 4**  
Untuk sembarang $i$ dan $k$, dengan $p = k - 1$ dan $s = 1$:

$$
o = i + 2(k - 1) - (k - 1) = i + (k - 1)
$$
![stride](../Asset/Full-Padding.png)

Biasanya, konvolusi membuat output lebih kecil dari input, karena kernel tidak bisa lewat di luar batas gambar.
Tapi dengan full padding, kita tambahkan nol di sekeliling input agar kernel bisa meliputi semua kemungkinan posisi, termasuk yang hanya menyentuh sebagian area input.


**Intuisi:**

| Jenis Padding | Rumus Output | Efek |
|----------------|---------------|-------|
| Tanpa padding | $o = i - k + 1$ | Output mengecil |
| Half / Same padding | $o = i$ | Output sama |
| Full padding | $o = i + k - 1$ | Output membesar |

### 2.3 No zero padding, non-unit strides
Kasus:

$p$ = 0 (tidak ber-padding) dan $s$ > 1 (stride lebih dari 1)

Stride > 1 artinya kernel melompat beberapa piksel sekaligus saat bergerak.
Jadi, tidak semua posisi input dikunjungi oleh kernel → hasilnya output jadi lebih kecil.

**Hubungan 5**

$$
o = \left\lfloor \frac{i - k}{s} \right\rfloor + 1
$$
![stride](../Asset/No-zero-padding,non-unit-strides.png)

Kenapa kita memakai floor (pembulatan kebawah)? 
Karena kadang kernel tidak pas berhenti di ujung input. Kalau stride-nya besar, sisa beberapa piksel di ujung bisa tidak ter-cover.

### 2.4 Zero padding, non-unit strides
Kasus:

$p$ > 0 dan $s$ > 1 (stride lebih dari 1)

hubungan 5 sudah menangani stride > 1 tetapi tanpa padding, kalau kita ingin menambahkan padding, kita tinggal mengganti ukuran input efektif dari $i$ menjadi $i + 2_p$. Ini adalah trik  yang sama persis dilakukan waktu dari hubungan 1 → hubungan 2 (waktu stride = 1)

**Hubungan 6**
$$
o = \left\lfloor \frac{i + 2p - k}{s} \right\rfloor + 1
$$

![stride](../Asset/Zero-padding,non-unit-strides.png)

Akan tetapi penggunaan fungsi floor dalam rumus dimensi output konvolusi, dan ini hanya berlaku untuk non-unit strides ($s > 1$). Ini merupakan inti dari Ambiguitas Dimensi Output dalam operasi konvolusi pada hubungan 6

![stride](../Asset/Ambiguitas.png)

Lihat kan, ukuran input yang berbeda menghasilkan hasil konvolusinya menjadi ukuran output yang sama

intinya Fungsi floor yang pada dasarnya adalah pembulatan ke bawah. Ia membuang sisa atau pecahan dari hasil pembagian. misalnya nilai awal (pembilang) berbeda ($4$ dan $5$), setelah dibagi 2 dan dibulatkan ke bawah, hasilnya sama (2).

## Chapter 3 Pooling arithmetic

Dalam jaringan saraf, lapisan pooling memberikan invariansi terhadap pergeseran kecil (small translations) pada input. Pooling yang paling sering digunakan adalah max pooling, yaitu, Membagi input menjadi beberapa patch kecil (biasanya tidak saling tumpang tindih / non-overlapping), lalu Mengambil nilai maksimum dari setiap patch untuk menjadi output.

Atau Mean pooling atau average pooling, yang mengambil rata-rata nilai dari patch, bukan maksimum. Intinya Semua jenis pooling ini punya prinsip dasar yang sama, Menggabungkan (meng-aggregate) nilai lokal dari sebagian kecil area input dengan fungsi non-linear tertentu.

semua hubungan matematis (relationship) yang diturunkan untuk konvolusi di bab sebelumnya juga berlaku untuk pooling — kecuali:

Pooling tidak menggunakan zero padding, jadi rumusnya lebih sederhana.

$$
o = \left\lfloor \frac{i - k}{s} \right\rfloor + 1
$$

## Chapter 4 Transposed Convolution Arithmetic

Transposed convolution, yang kadang juga disebut deconvolution, muncul karena kita ingin melakukan operasi yang secara arah berlawanan dengan konvolusi biasa. Kalau pada konvolusi standar kita mengubah sebuah citra atau fitur berukuran besar menjadi representasi yang lebih kecil dan padat, maka pada transposed convolution kita ingin melakukan kebalikannya — yaitu memproyeksikan representasi kecil tersebut kembali menjadi sesuatu yang lebih besar, menyerupai bentuk aslinya.

Kebutuhan ini sering muncul dalam berbagai konteks. Misalnya, pada arsitektur convolutional autoencoder, bagian encoder bertugas menyusutkan dimensi input menggunakan konvolusi biasa. Sebaliknya, bagian decoder harus memperbesar kembali hasil penyusutan itu agar bisa merekonstruksi citra aslinya. Operasi yang digunakan untuk memperbesar inilah yang disebut transposed convolution. Hal yang sama juga ditemukan pada model Generative Adversarial Network (GAN), terutama pada generator yang bertugas membuat gambar dari representasi acak berukuran kecil.

Jika dibandingkan dengan lapisan fully-connected (atau dense layer), transposed convolution punya dasar konsep yang serupa. Dalam lapisan fully-connected, operasi maju ditulis sebagai $y$ = $W_x$, dimana W adalah matriks bobot. Untuk membalikkan arah transformasi, kita bisa menggunakan $W^T$, yaitu matriks yang ditranspos. Maka pada dasarnya, transposed convolution juga melakukan hal yang sama — hanya saja dalam bentuk operasi konvolusi. Namun, karena konvolusi bekerja dengan cara sliding window pada data spasial (misalnya gambar), proses “transpose” ini tidak sesederhana sekadar membalikkan matriks.

Intinya, transposed convolution adalah cara untuk memperluas representasi spasial dengan pola konektivitas yang tetap konsisten dengan konvolusi biasa. Ia tidak secara matematis merupakan “invers” dari konvolusi, tetapi secara struktural ia membalik arah transformasi: dari ruang fitur yang padat menuju ruang yang lebih besar

### 4.1 Convolution as a matrix operation
operasi konvolusi—yang secara konseptual adalah proses pergeseran kernel—dapat diwakili (direpresentasikan) sebagai perkalian matriks linear dalam aljabar.

Representasi ini penting karena memungkinkan kita menggunakan teknik aljabar matriks standar, seperti transpose, untuk menghitung backward pass (propagasi balik kesalahan), yang merupakan inti dari pelatihan jaringan saraf.

Sebagai contoh, ambil operasi konvolusi yang digambarkan pada Gambar 2.1.
Jika input dan output diratakan menjadi vektor (dibaca dari kiri ke kanan, atas ke bawah),
maka konvolusi dapat direpresentasikan sebagai sebuah matriks jarang (sparse matrix) $C$, dimana elemen-elemen tak-nolnya adalah elemen-elemen $w_{i,j}$
![Convolution as a matrix operation](../Asset/Conv-as-matrix-operation.png)

dengan matriks $C$ menjadi
$$
\begin{pmatrix}
w_{0,0} & w_{0,1} & w_{0,2} & 0 & w_{1,0} & w_{1,1} & w_{1,2} & 0 & w_{2,0} & w_{2,1} & w_{2,2} & 0 & 0 & 0 & 0 & 0 \\
0 & w_{0,0} & w_{0,1} & w_{0,2} & 0 & w_{1,0} & w_{1,1} & w_{1,2} & 0 & w_{2,0} & w_{2,1} & w_{2,2} & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & w_{0,0} & w_{0,1} & w_{0,2} & 0 & w_{1,0} & w_{1,1} & w_{1,2} & 0 & w_{2,0} & w_{2,1} & w_{2,2} & 0 \\
0 & 0 & 0 & 0 & 0 & w_{0,0} & w_{0,1} & w_{0,2} & 0 & w_{1,0} & w_{1,1} & w_{1,2} & 0 & w_{2,0} & w_{2,1} & w_{2,2}
\end{pmatrix}
$$

Operasi linear ini (perkalian matriks $\mathbf{C}$ dengan vektor input) mengambil matriks input yang diratakan menjadi vektor 16-dimensi ( input awal $4 \times 4$ diubah menjadi vektor kolom $16 \times 1$) dan menghasilkan vektor 4-dimensi ($\mathbf{y} = \mathbf{C} \mathbf{x}$ atau $\mathbf{y}_{4 \times 1} = \mathbf{C}_{4 \times 16} \mathbf{x}_{16 \times 1}$) yang kemudian dibentuk ulang (reshaped) menjadi matriks output $2 \times 2$

Menggunakan representasi ini, alur mundur (backward pass) dapat dengan mudah diperoleh dengan mentranspos (transpose) $\mathbf{C}$; dengan kata lain, kesalahan dipropagasi balik (backpropagated) dengan mengalikan kerugian dengan $\mathbf{C}^{\text{T}}$. Operasi ini mengambil vektor 4-dimensi sebagai input dan menghasilkan vektor 16-dimensi sebagai output, dan pola konektivitasnya kompatibel dengan $\mathbf{C}$ berdasarkan konstruksi.


Secara singkat, matriks $\mathbf{C}^{\text{T}}$ berfungsi sebagai mekanisme untuk menyebarkan loss dari output $y$ kembali ke input $x$.

Ada tiga alasan utama mengapa representasi ini sangat penting:

1. Memungkinkan Propagasi Balik (Backpropagation)

Matriks C Transpose ($\mathbf{C}^{\text{T}}$): Representasi konvolusi sebagai perkalian matriks $\mathbf{C}$ memungkinkan kita menggunakan transpose $\mathbf{C}^{\text{T}}$ untuk menghitung gradien kesalahan yang harus mengalir mundur (backward pass).

$$  \frac{\partial L}{\partial x} = \mathbf{C}^{\text{T}} \frac{\partial L}{\partial y}$$

$\mathbf{C}^{\text{T}}$ Tanpa representasi aljabar linear ini, menghitung turunan (gradien) dari operasi konvolusi yang kompleks (melibatkan pergeseran kernel dan penjumlahan) akan sangat sulit dan tidak standar. Representasi matriks mengubahnya menjadi masalah aljabar linear standar.

2. Implementasi Komputasi yang Efisien

Meskipun secara konseptual kita memvisualisasikan konvolusi sebagai kernel yang bergeser, di belakang, implementasi komputasi modern mengandalkan perkalian matriks karena alasan efisiensi perangkat keras:

- Pustaka Aljabar Linear: Pustaka matematika seperti BLAS (Basic Linear Algebra Subprograms) dan cuBLAS (untuk GPU) dioptimalkan untuk melakukan perkalian matriks.

- Perangkat Keras: GPU (Graphics Processing Unit) dirancang khusus untuk memproses operasi matriks besar secara paralel. Dengan mengubah konvolusi menjadi perkalian matriks besar ($\mathbf{C} \mathbf{x}$), kita bisa memanfaatkan kecepatan dari si GPU, yang membuat pelatihan CNN menjadi lebih cepat.

3. Memahami Transposed Convolution (bab abis ini)

Representasi matriks adalah dasar untuk memahami Transposed Convolution.

- Definisi Operasi Transpose: Konvolusi Transpose didefinisikan secara aljabar sebagai operasi yang menggunakan $\mathbf{C}^{\text{T}}$ sebagai operator forward pass untuk melakukan up-sampling (memperbesar dimensi).

- Jika misalnya kita tidak tahu tentang $\mathbf{C}$ dan $\mathbf{C}^{\text{T}}$, kita tidak akan bisa secara matematis menjelaskan mengapa Transposed Convolution ($2 \times 2 \to 4 \times 4$) adalah kebalikan yang benar dari Konvolusi Langsung ($4 \times 4 \to 2 \times 2$).


### 4.2 Transposed Convolution
Konvolusi Transpose (sering disebut fractionally strided convolution) adalah mekanisme yang digunakan untuk meningkatkan dimensi spasial dari peta fitur (up-sampling) sambil mempertahankan pola konektivitas yang dipelajari dari kernel.

Inti dari Konvolusi Transpose adalah bahwa satu kernel ($w$) yang sama dapat mendefinisikan dua operasi yang berbeda secara fungsional: Konvolusi Langsung dan Konvolusi Transpose.

- Matriks Konvolusi ($\mathbf{C}$): Kernel $w$ membentuk matriks $\mathbf{C}$, yang digunakan untuk operasi Down-sampling (mengurangi ukuran).

- Matriks Transpose ($\mathbf{C}^{\text{T}}$): Transpose dari matriks ini ($\mathbf{C}^{\text{T}}$) adalah operator untuk operasi Up-sampling (memperbesar ukuran).

Penukaran Alur (Pass Swapping)

Transposed Convolution bekerja dengan menukar alur maju (forward pass) dan alur mundur (backward pass) dari konvolusi langsung (biasa). Kernel ($w$) yang sama mendefinisikan kedua operasi tersebut. Yang menentukan jenis operasinya adalah operator matriks mana yang digunakan untuk forward dan backward pass.

| Operasi                | Forward Pass (Input → Output)                     | Backward Pass (Error → Input)                                  |
|-------------------------|--------------------------------------------------|----------------------------------------------------------------|
| **Konvolusi Langsung** | Perkalian dengan matriks $\mathbf{C}$            | Perkalian dengan matriks $\mathbf{C}^{\text{T}}$               |
| **Konvolusi Transpose**| Perkalian dengan matriks $\mathbf{C}^{\text{T}}$ | Perkalian dengan matriks $(\mathbf{C}^{\text{T}})^{\text{T}} = \mathbf{C}$ |

Emulasi dengan Konvolusi LangsungSecara matematis, selalu mungkin untuk meniru (emulate) Transposed Convolution menggunakan Konvolusi Langsung. Namun, metode ini melibatkan penambahan banyak kolom dan baris nol (zero padding) pada input, yang menghasilkan implementasi yang jauh kurang efisien. Karena alasan inilah Transposed Convolution (menggunakan $\mathbf{C}^{\text{T}}$) lebih disukai.

### 4.3 No zero padding, unit strides, transposed
Sekarang kita bahas kasus Transposed Convolution paling sederhana: yang berasal dari Konvolusi Langsung tanpa zero padding ($p=0$) dan menggunakan unit strides ($s=1$).

Cara termudah untuk memahami Transposed Convolution adalah dengan memandangnya sebagai operasi yang memulihkan bentuk spasial dari peta fitur input asli. **Konvolusi Langsung**: Mengurangi dimensi spasial (misalnya, $4 \times 4 \to 2 \times 2$). **Transposed Convolution**: Berusaha mengembalikan dimensi spasial ke ukuran awal (misalnya, $2 \times 2 \to 4 \times 4$).

Sekarang kita pertimbangkan konvolusi kernel $3 \times 3$ pada input $4 \times 4$ menggunakan unit stride (langkah satu) dan tanpa padding (yaitu, $i=4$, $k=3$, $s=1$, dan $p=0$). Seperti yang digambarkan pada Gambar 2.1, ini menghasilkan output $2 \times 2$. Transpose dari konvolusi ini kemudian akan memiliki output berbentuk $4 \times 4$ ketika diterapkan pada input $2 \times 2$.

Cara lain untuk mendapatkan hasil dari transposed convolution adalah dengan menerapkan direct convolution (konvolusi langsung) yang setara—namun jauh kurang efisien. Contoh yang dijelaskan sejauh ini dapat diatasi dengan mengkonvolusikan kernel $3 \times 3$ pada input $2 \times 2$ yang di-pad dengan batas nol $2 \times 2$ menggunakan unit strides (yaitu, $i'=2$ [input baru], $k'=k$ [kernel tetap], $s'=1$ [stride tetap], dan $p'=2$ [padding baru]), seperti yang ditunjukkan pada Gambar 4.1. Yang menarik, ukuran kernel dan stride tetap sama, tetapi input dari transposed convolution sekarang diberi zero padding.

| Operasi                      | Parameter                     | Dimensi Input → Output |
|------------------------------|--------------------------------|------------------------|
| **Konvolusi Langsung (Asal)** | i=4, k=3, s=1, p=0             | 4×4 → 2×2              |
| **Transposed Convolution**    | i′=2, k′=3, s′=1               | 2×2 → 4×4              |
| **Emulasi Konvolusi Langsung**| i′=2, k′=3, s′=1, p′=2          | 2×2 → 4×4              |

intinya meskipun metode "emulasi" (peniruan) Transposed Convolution menggunakan Konvolusi Langsung (biasa). Untuk mencapai up-sampling $2 \times 2 \to 4 \times 4$ dengan konvolusi biasa, kita tetep harus secara artifisial menambahkan padding nol yang besar ($p=2$ atau $k-1$). Metode ini kurang efisien karena melibatkan perkalian banyak angka nol.

Salah satu cara untuk memahami logika di balik zero padding adalah dengan mempertimbangkan pola konektivitas dari konvolusi transpose dan menggunakannya untuk memandu desain konvolusi yang ekuivalen. Sebagai contoh, piksel kiri atas dari input konvolusi langsung hanya berkontribusi pada piksel kiri atas dari output, piksel kanan atas hanya terhubung ke piksel output kanan atas, dan seterusnya

Untuk mempertahankan pola konektivitas yang sama dalam konvolusi yang ekuivalen, diperlukan zero pad pada input sedemikian rupa sehingga aplikasi kernel yang pertama (kiri atas) hanya menyentuh piksel kiri atas, yaitu, padding harus sama dengan ukuran kernel dikurangi satu ($p' = k-1$)