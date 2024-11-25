# Laporan Proyek Machine Learning - Muhammad Abdiel Al Hafiz
## Domain Proyek
### Latar Belakang
Seiring dengan kemajuan teknologi dan inovasi yang terus berkembang, e-commerce telah menjadi bagian penting dalam kehidupan sehari-hari. Belanja online memberikan kemudahan bagi konsumen untuk mengakses berbagai produk hanya dengan beberapa klik (Referensi 1). Di industri fashion, e-commerce berkembang pesat karena kemudahan dalam membeli produk dari rumah tanpa harus mengunjungi toko fisik. Namun, tantangan utama yang dihadapi oleh konsumen adalah kesulitan dalam menemukan produk yang sesuai dengan preferensi mereka, seperti ukuran, warna, dan gaya, di tengah banyaknya pilihan produk yang tersedia di platform online (Referensi 3). Oleh karena itu, solusi yang dapat membantu pengguna menemukan produk yang relevan sangat dibutuhkan.

Dalam e-commerce, terutama di sektor fashion, navigasi yang sulit dan pencarian produk yang tidak efisien menjadi masalah utama. Konsumen sering kali harus menghabiskan waktu yang lama untuk menyaring ribuan produk untuk menemukan apa yang mereka inginkan, tanpa bantuan karyawan toko seperti di toko fisik (Referensi 2). Hal ini menambah waktu dan usaha yang dibutuhkan untuk membeli produk yang sesuai, dan sering kali menyebabkan frustrasi. Oleh karena itu, diperlukan sebuah sistem yang dapat menyaring produk-produk fashion yang relevan sesuai dengan preferensi individu pengguna.

Untuk mengatasi permasalahan tersebut, sistem rekomendasi berbasis teknologi machine learning dapat digunakan. Sistem ini membantu konsumen menemukan produk yang relevan dengan minat dan preferensi mereka secara cepat dan efisien (Referensi 1). Di sektor fashion, sistem rekomendasi dapat menyarankan produk berdasarkan karakteristik seperti kategori produk, ukuran, warna, atau bahkan preferensi yang pernah ditunjukkan oleh pengguna sebelumnya. Dengan mengimplementasikan sistem rekomendasi, platform e-commerce dapat meningkatkan peluang konsumen untuk menambahkan produk ke dalam keranjang belanja mereka dan akhirnya melakukan pembelian (Referensi 3).

Agar sistem rekomendasi dapat berfungsi dengan baik, data yang tepat perlu dikumpulkan dan diproses. Data tentang produk fashion seperti kategori, ukuran, warna, harga, dan merek, serta data pengguna seperti preferensi dan rating produk, sangat penting untuk menciptakan rekomendasi yang relevan dan personal (Referensi 3). Pengolahan data yang baik dan pemilihan data yang relevan sangat krusial agar sistem rekomendasi dapat memberikan hasil yang optimal bagi konsumen dan meningkatkan pengalaman berbelanja mereka.

Dengan penerapan sistem rekomendasi berbasis machine learning, platform e-commerce dapat membantu pengguna menemukan produk fashion yang tepat sesuai dengan preferensi mereka, sehingga meningkatkan kepuasan pelanggan dan konversi penjualan. Rekomendasi yang tepat akan meningkatkan efisiensi pencarian produk dan membantu konsumen menghemat waktu serta usaha dalam memilih produk yang sesuai dengan kebutuhan mereka (Referensi 2). Selain itu, platform e-commerce dapat memperoleh wawasan yang lebih dalam tentang preferensi pengguna dan mengoptimalkan penawaran produk mereka sesuai dengan permintaan pasar yang terus berkembang (Referensi 1).

## Business Understanding
Pembeli produk fashion di platform e-commerce adalah individu yang berusaha menemukan produk yang sesuai dengan preferensi mereka 
(misalnya, ukuran, warna, gaya). Dengan adanya banyak pilihan produk, konsumen membutuhkan cara yang efisien untuk menemukan produk yang relevan dan memenuhi kebutuhan mereka.

E-commerce fashion menghadapi tantangan dalam membantu konsumen menemukan produk yang relevan di antara ribuan pilihan. 
Konsumen sering kali menghabiskan banyak waktu untuk menelusuri produk secara manual, yang dapat menyebabkan frustasi. 
Rekomendasi yang tidak relevan atau sistem pencarian yang tidak efisien semakin memperburuk masalah ini. Oleh karena itu, dibutuhkan sistem yang dapat menyaring dan menampilkan produk yang relevan secara otomatis.

Solusi yang diusulkan adalah mengimplementasikan sistem rekomendasi berbasis machine learning pada platform e-commerce fashion. 
Dengan menggunakan data pengguna (misalnya, preferensi produk, rating, pencarian sebelumnya) dan data produk (misalnya, kategori, ukuran, warna, merek), 
sistem rekomendasi akan memberikan saran produk yang relevan. Hal ini akan membantu konsumen menghemat waktu, meningkatkan pengalaman berbelanja, dan pada akhirnya meningkatkan penjualan platform.

### Problem Statement
Berdasarkan kondisi yang telah diuraikan sebelumnya, proyek ini akan mengembangkan sistem rekomendasi produk fashion berbasis data. Sistem ini dirancang untuk membantu pengguna menemukan produk yang sesuai dengan preferensi mereka dengan memanfaatkan atribut produk dan interaksi pengguna.

Dengan menggunakan teknologi machine learning algoritma Content Based Filtering dan Collaborative Filtering diharapkan dapat menjawab permasalahan berikut:
- Bagaimana cara memanfaatkan data kategorikal seperti nama produk, brand, kategori, warna, dan ukuran untuk mengidentifikasi kemiripan produk sehingga dapat menghasilkan rekomendasi yang relevan dan membantu pengguna menemukan produk sesuai preferensi mereka?
- Bagaimana sistem dapat memanfaatkan data rating pengguna untuk menghasilkan rekomendasi produk yang belum pernah diberi rating, sehingga membantu pengguna menemukan produk baru yang sesuai dengan preferensinya?

### Goals
Untuk menjawab roblem statement tersebut, akan dibuat sistem rekomendasi dengan tujuan atau goals sebagai berikut:
- Mengembangkan sistem rekomendasi berbasis konten untuk menganalisis kemiripan produk berdasarkan atribut seperti nama produk, brand, kategori, warna, dan ukuran.
- Mengembangkan sistem rekomendasi berbasis kolaboratif untuk merekomendasikan produk yang belum pernah diberi rating oleh pengguna dengan memanfaatkan data rating pengguna sebelumnya.

### Solution Statement
Untuk mencapai goals tersebut, ada 2 pendekatan yang akan digunakan yaitu:
- **Content Based Filtering**: Untuk merekomendasikan produk berdasarkan atribut, metode content-based filtering akan menggunakan cosine similarity untuk menganalisis kemiripan antar produk berdasarkan fitur kategorikal seperti nama produk, brand, kategori, warna, dan ukuran. Cosine similarity dipilih karena mampu menghitung kemiripan antar produk secara efisien dengan mempertimbangkan vektor representasi atribut produk. Model ini akan dibuat dengan terlebih dahulu mengonversi data kategorikal menjadi representasi numerik melalui teknik seperti one-hot encoding atau TF-IDF. Kemudian, hasil kemiripan ini akan digunakan untuk memberikan rekomendasi produk yang serupa dengan preferensi pengguna.
- **Collaborative Filtering dengan Neural Collaborative Filtering (NCF)**: Pendekatan ini akan menggunakan data rating pengguna untuk merekomendasikan produk yang mungkin diminati pengguna. NCF memanfaatkan teknik deep learning untuk menggabungkan representasi laten pengguna dan produk guna menangkap hubungan di antara keduanya.

## Data Understanding
Data yang akan digunakan pada proyek ini adalah dataset [Fashion Products](https://www.kaggle.com/datasets/bhanupratapbiswas/fashion-products/data) yang diunduh dari kaggle.

Dataset tersebut berisi 1000 records dan 9 kolom yang memiliki karakteristik sebagai berikut:
- User ID: ID unik yang mewakili setiap pengguna dalam dataset. Tidak mengandung informasi langsung terkait atribut pengguna.
- Product ID: ID unik yang mengidentifikasi produk tertentu. Berguna untuk membedakan setiap produk di katalog.
- Product Name: Nama produk yang memberikan informasi deskriptif mengenai produk tertentu.
- Brand: Nama merek atau label produk.
- Category: Kategori atau jenis produk dalam katalog.
- Price: Harga produk dalam satuan mata uang dollar.
- Rating: Penilaian pengguna terhadap produk.
- Color: Warna dari produk.
- Size: Ukuran dari produk.

Untuk memahai data, selanjutnya akan dilakukan proses berikut:
### 1. Data Loading
Supaya isi dataset lebih mudah dipahami, kita perlu melakukan proses loading data terlebih dahulu dengan import library pandas untuk dapat membaca file datanya.

### 2. Exploratory Data Analysis
#### Informasi Dataset
Mengecek informasi pada dataset dengan fungsi info() berikut.
<!-- ![image](https://github.com/user-attachments/assets/7add242c-e067-4bc0-b0ab-6601ec408017) -->

<img src="https://github.com/user-attachments/assets/7add242c-e067-4bc0-b0ab-6601ec408017" alt="image" width="300"/>

<br>

Berdasarkan informasi di atas, dataset ini memiliki beberapa kriteria antara lain:
- 1 Kolom dengan tipe float64 yaitu Rating
- 3 Kolom dengan tipe int64 yaitu User ID, Product ID, dan Price
- 5 Kolom dengan tipe object yaitu Product Name, Brand, Category, Color, dan Size

#### Cek Missing Value
Jika data terdiri dari ratusan bahkan ribuan baris tentu akan susah dalam menemukan nilai field yang kosong. Oleh karena itu, Pandas memungkinkan kita dapat menemukan missing value secara cepat dengan fungsi isnull() dan sum().

<!-- ![image](https://github.com/user-attachments/assets/e2b21244-631d-477d-bab6-fcc763a4e689) -->

<img src="https://github.com/user-attachments/assets/e2b21244-631d-477d-bab6-fcc763a4e689" alt="image" width="150"/>

<br>

Berdaarkan informasi di atas tidak ditemukan adanya missing value, sehingga bisa dilanjutkan ke proses berikutnya.

#### Distribusi Produk berdasarkan Kategori
<!-- ![image](https://github.com/user-attachments/assets/6674fc20-a30d-4256-b44c-bdcf2b6d301c) -->

<img src="https://github.com/user-attachments/assets/6674fc20-a30d-4256-b44c-bdcf2b6d301c" alt="image" width="600"/>

<br>

Dari visualisasi di atas nampak beberapa informasi di antaranya:
- Secara keseluruhan ketiga kategori produk memiliki jumlah yang mirip.
- Kategori Kid's Fashion memiliki jumlah yang paling unggul disusul oleh Women's Fashion kemudian Men's Fashion.

#### Distribusi Produk berdasarkan Brand
<!-- ![image](https://github.com/user-attachments/assets/1ffa7e19-2530-4693-9a84-0830260b0ede) -->

<img src="https://github.com/user-attachments/assets/1ffa7e19-2530-4693-9a84-0830260b0ede" alt="image" width="600"/>

<br>

Dari visualasi di atas, didapatkan informasi sebagai berikut:
- Secara keseluruhan kelima brand memiliki jumlah produk yang tidak terlalu berbeda jauh.
- Brand Nike memiliki jumlah produk yang paling unggul dibandingkan brand yang lain.

#### Distribusi Brand berdasarkan Rating Tinggi (4 ke Atas) dan Rating Rendah (2 ke Bawah)
<!-- ![image](https://github.com/user-attachments/assets/b2141f15-cbb9-407c-843a-548327c4efa5) -->

<img src="https://github.com/user-attachments/assets/b2141f15-cbb9-407c-843a-548327c4efa5" alt="image" width="800"/>

<br>

Dari visualisasi di atas didapatkan beberapa informasi antara lain:
- Gucci unggul dalam produk dengan rating yang tinggi dan memiliki jumlah produk dengan rating rendah paling sedikit.
- Sebaliknya, Adidas justru memiliki rating yang rendah terbanyak dan memiliki paling sedikit produk dengan rating tinggi.
- Zara memiliki distribusi yang cukup seimbang antara jumlah produk dengan rating tinggi dan rating yang rendah.

#### Korelasi antara Harga dan Rating
<!-- ![image](https://github.com/user-attachments/assets/ff98b065-7f60-470d-8f80-de9278543f84) -->

<img src="https://github.com/user-attachments/assets/ff98b065-7f60-470d-8f80-de9278543f84" alt="image" width="600"/>

<br>

Terlihat bahwa horelasi di antara harga dan rating sebesar 0.0339 menunjukkan hubungan yang sangat lemah dan hampir tidak ada antara harga dan rating produk. Artinya, perubahasan harga tidak banyak berpengaruh pada penilaian pengguna.

## Data Preparation
Pada bagian ini akan dilakukan tahapan persiapan data yaitu menggabungkan kolom kategorikal.
Proses ini menggabungkan informasi dari beberapa kolom seperti Product Name, Brand, Category, Color, dan Size ke dalam kolom baru bernama description untuk menyediakan representasi teks terpadu dari setiap produk. Data kosong diisi dengan string kosong ('') untuk menghindari error, dan produk dengan deskripsi kosong sepenuhnya dihapus. Tujuannya adalah memastikan setiap produk memiliki deskripsi lengkap untuk analisis lebih lanjut.
```python
product_data['description'] = (
    product_data['Product Name'].fillna('') + " " +
    product_data['Brand'].fillna('') + " " +
    product_data['Category'].fillna('') + " " +
    product_data['Color'].fillna('') + " " +
    product_data['Size'].fillna('')
)

product_data = product_data[product_data['description'].str.strip() != '']
```

Dari proses ini didapatkan kolom description sebagai berikut:
<!-- ![image](https://github.com/user-attachments/assets/5b424e6f-0f83-4c62-9791-8e7f0f1d969b) -->

<img src="https://github.com/user-attachments/assets/5b424e6f-0f83-4c62-9791-8e7f0f1d969b" alt="image" width="800"/>

## Modeling
Pada proyek ini, model yang digunakan adalah Cosine Similarity untuk pendekatan content-based filtering dan Neural Collaborative Filtering (NCF) untuk pendekatan collaborative filtering. Cosine Similarity akan digunakan untuk mengukur kemiripan antar produk berdasarkan fitur kategorikal seperti nama, merek, kategori, warna, dan ukuran, sedangkan NCF akan memanfaatkan data rating pengguna untuk merekomendasikan produk yang belum pernah diberi rating oleh pengguna.

### Cosine Similarity
Cosine similarity adalah metode yang digunakan untuk mengukur tingkat kesamaan antara dua vektor dalam ruang multidimensi. Metode ini menghitung nilai kosinus dari sudut antara dua vektor, yang masing-masing direpresentasikan sebagai titik dalam ruang tersebut. Nilai cosine similarity berkisar dari -1 hingga 1, di mana nilai 1 menunjukkan kedua vektor sangat mirip (sepenuhnya sejajar), nilai 0 berarti tidak ada hubungan (tegak lurus), dan nilai -1 menunjukkan kedua vektor saling berlawanan (sepenuhnya tidak mirip). Metode ini sering diaplikasikan dalam analisis teks dan pengelompokan data untuk mengukur kesamaan antar dokumen atau fitur dalam suatu dataset.

```math
    \text{Cosine Similarity} = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \times \|\vec{B}\|}
```

Di mana:  
```math
\begin{aligned}
    \vec{A} & = \text{Vektor pertama, merepresentasikan satu entitas data (misalnya, produk atau pengguna).} \\
    \vec{B} & = \text{Vektor kedua, merepresentasikan entitas data lainnya.} \\
    \vec{A} \cdot \vec{B} & = \text{Hasil dot product antara } \vec{A} \text{ dan } \vec{B}, \; \textit{diperoleh dengan menjumlahkan perkalian elemen-elemen yang sesuai.} \\
    \|\vec{A}\| & = \text{Magnitudo (panjang) vektor } \vec{A}, \; \textit{dihitung sebagai akar kuadrat dari jumlah kuadrat elemen-elemen } \vec{A}. \\
    \|\vec{B}\| & = \text{Magnitudo (panjang) vektor } \vec{B}, \; \textit{dihitung serupa seperti } \|\vec{A}\|. \\
    \text{Cosine Similarity} & = \textit{Nilai yang menunjukkan tingkat kesamaan antara } \vec{A} \text{ dan } \vec{B}, \; \textit{berkisar antara -1 (berlawanan arah) hingga 1 (sejajar).}
\end{aligned}
```

Kemudian dilakukan pengujian model sebagai berikut:
#### Input Produk

| Product ID | Product Name | Brand | Category      | Color | Size |
|------------|--------------|-------|---------------|-------|------|
| 77         | Sweater      | H&M   | Kid's Fashion | Blue  | L    |

#### Hasil Rekomendasi
| Product ID | Product Name | Brand | Category      | Color | Size |
|------------|--------------|-------|---------------|-------|------|
| 687        | Sweater      | H&M   | Kid's Fashion | Blue  | L    |
| 765        | Sweater      | H&M   | Kid's Fashion | Blue  | S    |
| 361        | Sweater      | H&M   | Kids' Fashion | Blue  | XL   |
| 37         | Sweater      | H&M   | Kids' Fashion | Blue  | S    |
| 492        | Sweater      | H&M   | Kids' Fashion | Blue  | M    |


Berdasarkan hasil pengujian model Content-Based Filtering dengan menggunakan filter description, sistem berhasil merekomendasikan lima produk fashion yang serupa dengan produk ID 77. Produk-produk ini termasuk beberapa produk dari kategori yang sama, yaitu Kid's Fashion, serta brand yang sama, H&M. Hal ini menunjukkan bahwa jika seorang pengguna tertarik dengan produk seperti Sweater dari H&M dalam kategori Kid's Fashion dengan warna biru dan ukuran L, sistem dapat memberikan rekomendasi produk serupa, baik dalam hal kategori, brand, maupun deskripsi produk.

Dengan pendekatan ini, sistem menggunakan kemiripan dalam deskripsi produk untuk mengidentifikasi produk-produk lain yang mungkin menarik bagi pengguna berdasarkan preferensi mereka terhadap produk ID 77. Dalam hal ini, deskripsi produk menjadi fitur utama untuk menilai kesamaan produk, yang memungkinkan sistem memberikan rekomendasi yang relevan dan sesuai dengan apa yang disukai oleh pengguna.

Kelebihan Cosine Similarity:
- Independensi terhadap panjang vektor: Cosine similarity mengukur kesamaan berdasarkan arah vektor, bukan magnitudo. Ini berarti bahwa dua vektor yang memiliki panjang berbeda tetapi arah yang sama (misalnya, dua produk dengan deskripsi serupa) dapat dianggap sangat mirip.
- Efektif dalam teks dan data spars: Dalam aplikasi seperti pemrosesan teks atau sistem rekomendasi berbasis konten, cosine similarity sangat efektif karena sering kali data yang digunakan (misalnya, kata-kata dalam dokumen) sangat jarang (sparse), dan metode ini dapat menangani data tersebut dengan baik.
- Skalabilitas: Cosine similarity dapat diterapkan pada dataset besar tanpa memerlukan komputasi yang sangat berat, karena perhitungannya hanya bergantung pada produk titik dan panjang vektor yang dapat dihitung dengan efisien.

Kekurangan Cosine Similarity:
- Tidak memperhitungkan urutan atau konteks: Cosine similarity hanya mengukur kesamaan antara dua vektor, tanpa mempertimbangkan urutan atau konteks data (misalnya, urutan kata dalam kalimat atau urutan preferensi pengguna dalam sistem rekomendasi).
- Sensitif terhadap data noise: Cosine similarity dapat menghasilkan hasil yang tidak akurat jika data memiliki banyak noise atau jika ada kata-kata atau fitur yang tidak relevan dalam representasi vektor, karena setiap elemen vektor berkontribusi pada hasil akhirnya.
- Tidak menangkap hubungan non-linear: Cosine similarity mengasumsikan hubungan linier antar elemen vektor, sehingga tidak dapat menangkap hubungan kompleks atau non-linear dalam data, yang mungkin lebih relevan dalam beberapa aplikasi (misalnya, dalam analisis citra atau data yang lebih kompleks).

### Neural Collaborative Filtering (NCF)
Neural Collaborative Filtering (NCF) adalah metode yang menggabungkan teknik Collaborative Filtering tradisional dengan kemampuan neural networks untuk memodelkan interaksi pengguna dan produk dengan cara yang lebih kompleks dan non-linear. NCF bertujuan untuk memberikan rekomendasi berdasarkan interaksi antara pengguna dan produk, tanpa memerlukan eksplisit informasi tentang preferensi pengguna atau produk lainnya selain ID pengguna dan produk itu sendiri pada proyek ini.

#### Arsitektur NCF
##### 1. Embedding Layer
- User Embedding: Setiap pengguna diwakili dalam bentuk vektor berdimensi rendah (embedding). Proses embedding ini mereduksi ID pengguna yang dapat sangat besar (misalnya, jutaan ID pengguna) ke dalam ruang berdimensi lebih kecil yang menyimpan informasi relevan tentang preferensi pengguna.
- Product Embedding: Setiap produk juga diwakili dengan cara yang sama, yaitu vektor embedding, yang mengonversi ID produk menjadi representasi numerik berdimensi rendah.

```math
\vec{e_u} \in \mathbb{R}^d, \quad \vec{e_p} \in \mathbb{R}^d
```

Di mana:

```math
\begin{aligned}
    \vec{e_u} & = \text{Vektor embedding untuk pengguna (user),} \; \textit{diambil dari matriks embedding pengguna } W_u. \\
    \vec{e_p} & = \text{Vektor embedding untuk produk (item),} \; \textit{diambil dari matriks embedding produk } W_p. \\
    d & = \text{Dimensi dari embedding, yaitu jumlah fitur yang digunakan untuk merepresentasikan setiap entitas.}
\end{aligned}
```

##### 2. Concatenation
Setelah proses embedding, vektor-vektor dari pengguna dan produk digabungkan (concatenate) menjadi satu vektor panjang yang menggabungkan informasi kedua entitas tersebut. Dengan kata lain, pengguna dan produk saling berinteraksi dalam satu ruang yang lebih luas.


```math
\vec{e_u} \oplus \vec{e_p} \in \mathbb{R}^{2d}
```

Di mana:

```math
\begin{aligned}
    \vec{e_u} & = \text{Vektor embedding untuk pengguna (user),} \; \textit{diambil dari matriks embedding pengguna } W_u. \\
    \vec{e_p} & = \text{Vektor embedding untuk produk (item),} \; \textit{diambil dari matriks embedding produk } W_p. \\
    \oplus & = \text{Operasi concatenation, yaitu menggabungkan kedua vektor menjadi satu vektor.} \\
    2d & = \text{Dimensi vektor hasil concatenation, yaitu dua kali dimensi embedding (\(d\)) karena kita menggabungkan dua vektor.}
\end{aligned}
```

##### 3. Multilayer Perceptron (MLP)
Vektor gabungan yang diperoleh dari concatenation kemudian diproses melalui beberapa lapisan jaringan syaraf (MLP) untuk menangkap interaksi yang lebih kompleks antara pengguna dan produk. Setiap lapisan dalam MLP memiliki unit tersembunyi yang mengaktifkan (activation) hubungan non-linear antara input dan output. Misalnya, lapisan pertama bisa berukuran 128 unit, dan lapisan kedua bisa berukuran 64 unit.

```math
\text{MLP}\left( \vec{e_u} \oplus \vec{e_p} \right)
```

Di mana:

```math
\begin{aligned}
    \vec{e_u} \oplus \vec{e_p} & = \text{Vektor hasil concatenation antara embedding pengguna dan produk,} \; \textit{seperti yang telah dijelaskan sebelumnya.} \\
    \text{MLP} & = \text{Fungsi Multi-Layer Perceptron, yang terdiri dari beberapa lapisan fully connected (dense) yang digunakan untuk memodelkan interaksi non-linear antara input.} \\
    \text{MLP}\left( \vec{e_u} \oplus \vec{e_p} \right) & = \text{Hasil output dari MLP yang memproses vektor concatenation, menghasilkan prediksi nilai akhir (misalnya rating atau relevansi).}
\end{aligned}
```

##### 4. Output Layer
Setelah melewati lapisan-lapisan MLP, output dari model adalah prediksi rating yang akan diberikan oleh pengguna terhadap produk. Fungsi aktivasi linear digunakan pada output layer untuk menghasilkan rating yang dapat berupa nilai kontinu.

```math
\hat{r}_{up} = f_{\text{MLP}}\left( \vec{e_u} \oplus \vec{e_p} \right)
```

Di mana:

```math
\begin{aligned}
    \hat{r}_{up} & = \text{Prediksi output untuk interaksi pengguna dan produk, misalnya rating atau relevansi, yang dihasilkan dari fungsi MLP.} \\
    f_{\text{MLP}} & = \text{Fungsi yang mengacu pada hasil dari proses Multi-Layer Perceptron, yang memetakan input (concatenation dari embedding pengguna dan produk) menjadi output akhir.} \\
    \vec{e_u} \oplus \vec{e_p} & = \text{Vektor hasil concatenation antara embedding pengguna dan produk, yang telah diproses oleh MLP untuk menghasilkan output.}
\end{aligned}
```

Kelebihan Neural Collaborative Filtering:
- Kemampuan Menangkap Pola Kompleks: NCF dapat menangkap pola yang lebih kompleks dalam data dibandingkan dengan metode tradisional seperti matrix factorization, karena menggunakan neural networks yang dapat belajar dari interaksi non-linear antara fitur pengguna dan produk.
- Fleksibilitas dalam Arsitektur: NCF memungkinkan eksperimen dengan berbagai jenis arsitektur neural networks (MLP, CNN, RNN) yang dapat disesuaikan untuk berbagai jenis data dan aplikasi, meningkatkan kemampuannya untuk menangani berbagai jenis data.
- Kemampuan untuk Memperbaiki Kekurangan Model Sederhana: Berbeda dengan model berbasis matriks atau nearest neighbors, NCF dapat mengatasi keterbatasan teknik-teknik tersebut dalam menangkap hubungan non-linear antara pengguna dan item, menjadikannya lebih efektif dalam prediksi rekomendasi.

Kekurangan Neural Collaborative Filtering:
- Kompleksitas dalam Pelatihan: Karena melibatkan deep learning, NCF membutuhkan lebih banyak data dan waktu untuk pelatihan dibandingkan dengan model berbasis algoritma sederhana, seperti collaborative filtering berbasis cosine similarity atau k-NN.
- Overfitting: NCF memiliki potensi lebih tinggi untuk overfitting, terutama jika data training tidak cukup besar atau jika model terlalu rumit, karena jaringan neural bisa belajar noise dalam data.
- Memerlukan Sumber Daya Komputasi yang Lebih Besar: Model NCF biasanya lebih mahal dalam hal komputasi, membutuhkan hardware yang lebih kuat (seperti GPU) untuk melatih model dalam waktu yang efisien, yang mungkin menjadi kendala di lingkungan dengan sumber daya terbatas.

Berikut merupakan hasil pengujian model Neural Collaborative Filtering
#### Produk yang diberi Rating oleh User

