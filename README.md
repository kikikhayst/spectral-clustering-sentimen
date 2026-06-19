# spectral-clustering-sentimen
Berikut adalah source code analisis sentimen dengan spectral clustering

## CARA MENJALANKAN PROGRAM CLUSTERING SENTIMEN

### Persiapan Lingkungan Pengembangan
1. Pastikan **Visual Studio Code** telah terinstal di perangkat Anda.
2. Unduh dan instal Bahasa pemrograman **Python**.
### Clone atau Unduh Proyek
1. Clone repositori aplikasi dari GitHub … atau unduh file ZIP proyek.
2. Buka Visual Studio Code, lalu pilih **Open Folder…*.
3. Arahkan ke direktori tempat proyek disimpan dan pilih folder proyek aplikasi bernama *dev*.
### Konfigurasi Proyek
1. Pastikan virtual environment `venv` telah terinstal. Atau instal environment pada terminal dengan ketik:
```
python -m venv env
```
2. Posisikan focus folder di terminal pada `./dev` (`.` Adalah direktori menyimpan folder dev yang berisi program)
. Kemudian pada terminal ketik:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
digunakan untuk mengubah kebijakan eksekusi skrip PowerShell pada sesi (process) yang sedang berjalan.
4. Instal module atau library yang dibutuhkan pada `requirements.txt` dengan ketik pada terminal:
```
pip install -r requirements.txt
```
### Jalankan Program (BELUM DIUBAH)
1. Buka terminal pada Visual Studio Code kemudian ketikkan:
`python main.py`
2. Tunggu hingga proses running selesai, akan muncul beberapa tampilan wordcloud yang telah dibuat oleh program.
3. Bila akan menambah tautan produk (hanya dari tokopedia), langkah-langkahnya yaitu:
    1. Buka produk yang akan dimasukkan,
    2. Scroll kebawah sedikit akan muncul tab **Ulasan**, klik tab tersebut,
    3. Kemudian scroll kebawah sampai menemukan tombol **Lihat Semua Ulasan**, Klik tombol tersebut,
    4. Kemudian lihat pada link apakah pada akhir link sudah berubah menjadi `/review`. Contoh: [https://www.tokopedia.com/scoraofficial/scora-sheer-glow-tone-up-cream-30-gr-tone-up-viral-mencerahkan-secara-natural-1730831226036323854/review]
    5. Copy paste link tersebut ke dalam file `main.py` pada bagian variable `product_urls`
### Debugging dan Penyelesaian Masalah
1. Jika terjadi error saat run code, periksa hasil keluaran pada terminal.
2. Pastikan tidak ada error terkait konfigurasi yang salah.
3. Lakukan perbaikan sesuai dengan pesan error yang muncul.

Setelah semua langkah selesai, aplikasi sudah dapat digunakan dan diuji lebih lanjut. 🚀

