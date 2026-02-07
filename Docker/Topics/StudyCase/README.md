# Materi Docker - Day 2

## 1. Studi Kasus 1: Build Image dengan Dockerfile (Go – Hello World)

### 1.1. Struktur Project

```
hello-go/
├── main.go
└── Dockerfile
```

---

### 1.2. Kode Program (`main.go`)

Web server Go sederhana:

```go
package main

import (
	"fmt"
	"net/http"
)

func handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "Hello World")
}

func main() {
	http.HandleFunc("/", handler)
	http.ListenAndServe(":8080", nil)
}
```

---

### 1.3. Dockerfile

```dockerfile
# Base image Go
FROM golang:1.22-alpine

# Set working directory
WORKDIR /app

# Copy source code
COPY main.go .

# Build binary
RUN go build -o app

# Expose port
EXPOSE 8080

# Run app
CMD ["./app"]
```

---

### 1.4. Penjelasan Syntax Dockerfile

Berikut adalah penjelasan detail mengenai instruksi yang digunakan dalam Dockerfile di atas:

| Instruksi | Penjelasan | Contoh |
|-----------|------------|--------|
| **FROM** | Mendefinisikan **base image** yang akan digunakan. Setiap Dockerfile harus dimulai dengan instruksi ini. Base image biasanya berisi OS minimal (seperti Alpine, Ubuntu) atau runtime bahasa pemrograman (seperti Node, Python, Go). | `FROM golang:1.22-alpine` <br> (Menggunakan image Go versi 1.22 berbasis Alpine Linux yang ringan) |
| **WORKDIR** | Menentukan **direktori kerja** di dalam container. Semua perintah selanjutnya (COPY, RUN, CMD) akan dijalankan di dalam folder ini. Jika folder belum ada, Docker akan membuatnya. | `WORKDIR /app` <br> (Masuk ke folder `/app` di dalam container) |
| **COPY** | Menyalin file atau folder dari **host** (komputer kital) ke dalam **container**. | `COPY main.go .` <br> (Salin `main.go` dari host ke folder kerja saat ini di container) |
| **RUN** | Menjalankan perintah command line saat proses **build image**. Biasa digunakan untuk install dependency, build aplikasi, atau setup konfigurasi. | `RUN go build -o app` <br> (Compile kode Go menjadi binary bernama `app`) |
| **EXPOSE** | Memberi informasi bahwa container akan "mendengarkan" koneksi pada port tertentu. Ini bersifat dokumentasi, kita tetap harus melakukan mapping port saat `docker run`. | `EXPOSE 8080` <br> (Container menggunakan port 8080) |
| **CMD** | Perintah default yang dijalankan ketika **container dimulai**. Berbeda dengan RUN yang jalan saat build, CMD jalan saat runtime. Hanya boleh ada satu CMD di Dockerfile. | `CMD ["./app"]` <br> (Jalankan binary `app`) |

---

## 2. Build Docker Image

### 2.1. Perintah Build

Gunakan perintah `docker build` untuk membuat image dari Dockerfile.

```bash
docker build -t hello-go:1.0 .
```

**Penjelasan:**
- `-t hello-go:1.0` : Memberi nama (tag) image `hello-go` dengan versi `1.0`.
- `.` : Menunjukkan lokasi Dockerfile ada di direktori saat ini.

### 2.2. Verifikasi Image

Cek apakah image berhasil dibuat:

```bash
docker images
```

---

## 3. Menjalankan Container

### 3.1. Perintah Run

Jalankan container dari image yang baru dibuat:

```bash
docker run -d \
--name hello-go \
-p 8080:8080 \
hello-go:1.0
```

### 3.2. Akses Aplikasi

Buka browser atau gunakan curl:

```
http://localhost:8080
```

➡️ **Output:**

```
Hello World
```

---

## 4. Upload Image ke Docker Registry (Docker Hub)

### 4.1. Login ke Docker Hub

```bash
docker login
```

Masukkan:
* Username Docker Hub
* Password / Access Token

### 4.2. Tag Image Sesuai Format Docker Hub

Agar bisa di-push, nama image harus diawali dengan username Docker Hub.

Format:
```
username/nama-image:versi
```

Perintah:
```bash
docker tag hello-go:1.0 usernameDockerHub/hello-go:1.0
```

### 4.3. Push ke Docker Hub

Upload image ke registry:

```bash
docker push usernameDockerHub/hello-go:1.0
```

📌 **Sekarang image image bersifat publik (atau privat tergantung setting) dan bisa di-pull di server mana pun:**

```bash
docker pull usernameDockerHub/hello-go:1.0
```

---

## 5. Environment Variable di Docker (JavaScript App)

Environment Variable memungkinkan kita mengubah konfigurasi aplikasi tanpa harus mengubah kode atau rebuild image.

Pada studi kasus ini kita menggunakan aplikasi **Node.js**.

**Target Output:**
```
Hello {NAMA_ENV}
```

---

## 6. Project JavaScript (Node.js)

### 6.1. Struktur Project

```
hello-env-js/
├── index.js
└── Dockerfile
```

### 6.2. Kode Program (`index.js`)

```js
const http = require("http");

const PORT = 3000;
// Mengambil value dari Environment Variable "NAMA_ENV"
// Jika tidak ada, default ke "World"
const NAMA = process.env.NAMA_ENV || "World";

const server = http.createServer((req, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });
  res.end(`Hello ${NAMA}`);
});

server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

### 6.3. Dockerfile

```dockerfile
FROM node:20-alpine

WORKDIR /app

COPY index.js .

EXPOSE 3000

CMD ["node", "index.js"]
```

---

## 7. Build Image JavaScript

Build image dengan nama `hello-env-js`:

```bash
docker build -t hello-env-js:1.0 .
```

---

## 8. Menjalankan Container dengan Environment Variable

### 8.1. Cara 1: Menggunakan Flag `-e`

Kita menyuntikkan variable `NAMA_ENV` saat menjalankan container.

```bash
docker run -d \
--name hello-env \
-p 3000:3000 \
-e NAMA_ENV=Dzaky \
hello-env-js:1.0
```

**Akses:** `http://localhost:3000`

➡️ **Output:**
```
Hello Dzaky
```

### 8.2. Cara 2: Multiple Environment Variable

Kita bisa menambahkan banyak variable dengan multiple flag `-e`.

```bash
docker run -d \
-p 3000:3000 \
-e NAMA_ENV=Docker \
-e MODE=production \
hello-env-js:1.0
```

---

## 9. Ringkasan

### 9.1. Dockerfile
*   **Blueprint** untuk membuat image secara otomatis.
*   Menggunakan instruksi langkah-demi-langkah (`FROM`, `COPY`, `RUN`, `CMD`).
*   Memastikan konsistensi aplikasi di berbagai environment.

### 9.2. Docker Registry
*   Tempat penyimpanan dan distribusi image.
*   `docker tag` untuk menamai ulang image agar sesuai format registry.
*   `docker push` untuk upload, `docker pull` untuk download.

### 9.3. Environment Variable
*   Teknik konfigurasi dinamis.
*   Menggunakan flag `-e` saat `docker run`.
*   Aplikasi membaca variable dari `process.env` (Node.js), `os.Getenv` (Go), dll.
