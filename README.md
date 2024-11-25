# Laporan Proyek Machine Learning - Muhammad Abdiel Al Hafiz
## Domain Proyek
### Latar Belakang
Seiring dengan kemajuan teknologi dan inovasi yang terus berkembang, e-commerce telah menjadi bagian penting dalam kehidupan sehari-hari. Belanja online memberikan kemudahan bagi konsumen untuk mengakses berbagai produk hanya dengan beberapa klik [[1](https://www.academia.edu/86124217/Amazon_Product_Recommendation_System)]. Di industri fashion, e-commerce berkembang pesat karena kemudahan dalam membeli produk dari rumah tanpa harus mengunjungi toko fisik. Namun, tantangan utama yang dihadapi oleh konsumen adalah kesulitan dalam menemukan produk yang sesuai dengan preferensi mereka, seperti ukuran, warna, dan gaya, di tengah banyaknya pilihan produk yang tersedia di platform online [[3](https://www.academia.edu/82914123/Enhancing_E_Commerce_Applications_with_Machine_Learning_Recommendation_Systems)].

Dalam e-commerce, terutama di sektor fashion, navigasi yang sulit dan pencarian produk yang tidak efisien menjadi masalah utama. Konsumen sering kali harus menghabiskan waktu yang lama untuk menyaring ribuan produk untuk menemukan apa yang mereka inginkan, tanpa bantuan karyawan toko seperti di toko fisik [[2](https://www.academia.edu/96238561/ERS_Latent_Dirichlet_Allocation_Based_E_Commerce_Recommendation_System_Using_Deep_Neural_Network)]. Hal ini menambah waktu dan usaha yang dibutuhkan untuk membeli produk yang sesuai, dan sering kali menyebabkan frustrasi.

Untuk mengatasi permasalahan tersebut, sistem rekomendasi berbasis teknologi machine learning dapat digunakan. Sistem ini membantu konsumen menemukan produk yang relevan dengan minat dan preferensi mereka secara cepat dan efisien [[1](https://www.academia.edu/86124217/Amazon_Product_Recommendation_System)]. Di sektor fashion, sistem rekomendasi dapat menyarankan produk berdasarkan karakteristik seperti kategori produk, ukuran, warna, atau bahkan preferensi yang pernah ditunjukkan oleh pengguna sebelumnya. Dengan mengimplementasikan sistem rekomendasi, platform e-commerce dapat meningkatkan peluang konsumen untuk menambahkan produk ke dalam keranjang belanja mereka dan akhirnya melakukan pembelian [[3](https://www.academia.edu/82914123/Enhancing_E_Commerce_Applications_with_Machine_Learning_Recommendation_Systems)].

Agar sistem rekomendasi dapat berfungsi dengan baik, data yang tepat perlu dikumpulkan dan diproses. Data tentang produk fashion seperti kategori, ukuran, warna, harga, dan merek, serta data pengguna seperti preferensi dan rating produk, sangat penting untuk menciptakan rekomendasi yang relevan dan personal [[3](https://www.academia.edu/82914123/Enhancing_E_Commerce_Applications_with_Machine_Learning_Recommendation_Systems)]. Pengolahan data yang baik dan pemilihan data yang relevan sangat krusial agar sistem rekomendasi dapat memberikan hasil yang optimal bagi konsumen dan meningkatkan pengalaman berbelanja mereka.

Dengan penerapan sistem rekomendasi berbasis machine learning, platform e-commerce dapat membantu pengguna menemukan produk fashion yang tepat sesuai dengan preferensi mereka, sehingga meningkatkan kepuasan pelanggan dan konversi penjualan. Rekomendasi yang tepat akan meningkatkan efisiensi pencarian produk dan membantu konsumen menghemat waktu serta usaha dalam memilih produk yang sesuai dengan kebutuhan mereka [[2](https://www.academia.edu/96238561/ERS_Latent_Dirichlet_Allocation_Based_E_Commerce_Recommendation_System_Using_Deep_Neural_Network)]. Selain itu, platform e-commerce dapat memperoleh wawasan yang lebih dalam tentang preferensi pengguna dan mengoptimalkan penawaran produk mereka sesuai dengan permintaan pasar yang terus berkembang [[1](https://www.academia.edu/86124217/Amazon_Product_Recommendation_System)].

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
Pada proyek ini, model yang digunakan adalah Cosine Similarity untuk pendekatan content-based filtering dan Neural Collaborative Filtering (NCF) untuk pendekatan collaborative filtering. Cosine Similarity akan digunakan untuk mengukur kemiripan antar produk berdasarkan fitur kategorikal seperti nama, merek, kategori, warna, dan ukuran, sedangkan NCF akan memanfaatkan data rating pengguna untuk merekomendasikan produk yang belum pernah diberi rating oleh pengguna. Model yang sudah dilatih akan dievaluasi dengan metrik MSE dan RMSE, penjelasan lebih detail terkait metrik evaluasi akan dibahas saat evaluasi model.

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
| 77         | Sweater      | H&M   | Kids' Fashion | Blue  | L    |

#### Hasil Rekomendasi
| Product ID | Product Name | Brand | Category      | Color | Size |
|------------|--------------|-------|---------------|-------|------|
| 687        | Sweater      | H&M   | Kids' Fashion | Blue  | L    |
| 765        | Sweater      | H&M   | Kids' Fashion | Blue  | S    |
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
Berikut merupakan hasil pengujian model Neural Collaborative Filtering untuk user ID 98:
#### 5 Produk dengan Rating Tertinggi  oleh User
| Product ID | Product Name | Brand  | Category        | Price | Rating |
|------------|--------------|--------|-----------------|-------|--------|
| 29         | Shoes        | Gucci  | Women's Fashion | 85    | 4.94   |
| 898        | Dress        | Nike   | Women's Fashion | 44    | 4.21   |
| 779        | Jeans        | Nike   | Kids' Fashion   | 94    | 3.72   |
| 32         | T-shirt      | Nike   | Kids' Fashion   | 78    | 3.44   |
| 173        | Dress        | Nike   | Kids' Fashion   | 21    | 3.10   |

#### Top 5 Rekomendasi Produk
| Product ID | Product Name | Brand  | Category        | Price | Rating |
|------------|--------------|--------|-----------------|-------|--------|
| 33         | Jeans        | H&M    | Kid' Fashion    | 89    | 4.80   |
| 302        | T-shirt      | Gucci  | Kid' Fashion    | 37    | 4.90   |
| 616        | Dress        | Zara   | Women's Fashion | 98    | 4.87   |
| 701        | Shoes        | Gucci  | Kids' Fashion   | 58    | 3.80   |
| 704        | T-shirt      | Adidas | Kids' Fashion   | 38    | 3.98   |

Hasil rekomendasi model Neural Collaborative Filtering (NCF) untuk User ID 98 menunjukkan kemampuan model memahami preferensi historis pengguna. Produk yang diberi rating tinggi sebelumnya, seperti Shoes dari Gucci (Rating 4.94), menegaskan kecenderungan user terhadap merek premium. Mayoritas produk yang diberi rating berasal dari kategori Women's Fashion dan Kids' Fashion, dengan dominasi merek Nike, menunjukkan pola belanja yang berfokus pada merek dan kategori tertentu. Hal ini tercermin dalam rekomendasi model, yang menyarankan produk dari kategori Kids' Fashion (contohnya, Jeans dari H&M dan T-shirt dari Gucci) serta merek populer lainnya seperti Adidas dan Zara, yang relevan dengan preferensi user.

Prediksi rating untuk rekomendasi berkisar antara 3.80 hingga 4.90, mengindikasikan bahwa model berhasil mengidentifikasi produk yang berpotensi disukai pengguna berdasarkan embedding laten. Selain menyarankan produk serupa dengan preferensi sebelumnya, model juga mencoba mendiversifikasi rekomendasi dengan memasukkan produk dari kategori atau merek baru, seperti Women's Fashion dari Zara, untuk menawarkan lebih banyak variasi. Meskipun rekomendasi cukup relevan, potensi perbaikan meliputi pengayaan data dengan fitur tambahan seperti ulasan atau deskripsi produk serta penyesuaian sparsity untuk meningkatkan akurasi lebih lanjut.

Kelebihan Neural Collaborative Filtering:
- Kemampuan Menangkap Pola Kompleks: NCF dapat menangkap pola yang lebih kompleks dalam data dibandingkan dengan metode tradisional seperti matrix factorization, karena menggunakan neural networks yang dapat belajar dari interaksi non-linear antara fitur pengguna dan produk.
- Fleksibilitas dalam Arsitektur: NCF memungkinkan eksperimen dengan berbagai jenis arsitektur neural networks (MLP, CNN, RNN) yang dapat disesuaikan untuk berbagai jenis data dan aplikasi, meningkatkan kemampuannya untuk menangani berbagai jenis data.
- Kemampuan untuk Memperbaiki Kekurangan Model Sederhana: Berbeda dengan model berbasis matriks atau nearest neighbors, NCF dapat mengatasi keterbatasan teknik-teknik tersebut dalam menangkap hubungan non-linear antara pengguna dan item, menjadikannya lebih efektif dalam prediksi rekomendasi.

Kekurangan Neural Collaborative Filtering:
- Kompleksitas dalam Pelatihan: Karena melibatkan deep learning, NCF membutuhkan lebih banyak data dan waktu untuk pelatihan dibandingkan dengan model berbasis algoritma sederhana, seperti collaborative filtering berbasis cosine similarity atau k-NN.
- Overfitting: NCF memiliki potensi lebih tinggi untuk overfitting, terutama jika data training tidak cukup besar atau jika model terlalu rumit, karena jaringan neural bisa belajar noise dalam data.
- Memerlukan Sumber Daya Komputasi yang Lebih Besar: Model NCF biasanya lebih mahal dalam hal komputasi, membutuhkan hardware yang lebih kuat (seperti GPU) untuk melatih model dalam waktu yang efisien, yang mungkin menjadi kendala di lingkungan dengan sumber daya terbatas.

## Evaluation
Seperti yang telah dijelaskan sebelumnya, metrik evaluasi yang digunakan adalah Mean Square Error (MAE) dan Root Mean Square Error (RMSE). Kedua metrik tersebut dipilih karena keduanya memberikan ukuran yang jelas tentang sejauh mana prediksi rating atau preferensi produk yang diberikan oleh model menyimpang dari rating atau preferensi asli pengguna.

- Mean Squared Error (MSE)
    MSE menghitung rata-rata kuadrat dari selisih antara nilai prediksi dan nilai aktual.
    Dengan mengkuadratkan selisih, MSE memberikan bobot yang lebih besar pada kesalahan yang besar. Ini berarti bahwa model akan lebih "dihukum" jika membuat prediksi yang jauh dari nilai sebenarnya.
    
    MSE sering digunakan ketika kita ingin memberikan penalti yang lebih besar pada kesalahan yang besar, karena kesalahan yang besar dapat memiliki konsekuensi yang lebih signifikan.
    
    ```math
      \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
    ```
      
  Di mana:
  ```math
      \begin{aligned}
      \text{MSE} & = \text{Mean Squared Error} \\
      n & = \text{Jumlah data observasi} \\
      y_i & = \text{Nilai aktual ke-} i \\
      \hat{y}_i & = \text{Nilai prediksi ke-} i
      \end{aligned}
  ```

- Root Mean Squared Error (RMSE)
    RMSE (Root Mean Squared Error) adalah akar kuadrat dari MSE, yang memberikan ukuran rata-rata kesalahan prediksi dalam satuan yang sama dengan data asli.

    RMSE mempertahankan sifat MSE yang memberikan penalti lebih besar pada kesalahan besar, namun dengan membuat hasilnya lebih mudah dipahami karena mengembalikan kesalahan ke skala yang lebih mudah diinterpretasikan. RMSE sering digunakan ketika kita ingin menilai akurasi model dengan mempertimbangkan besarnya kesalahan dalam konteks yang lebih terukur dan dapat dibandingkan langsung dengan data aslinya.

    ```math
    \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( r_{i} - \hat{r}_{i} \right)^2}
    ```
    
    Di mana:
    
    ```math
    \begin{aligned}
        \text{RMSE} & = \text{Root Mean Squared Error, metrik untuk mengukur kesalahan prediksi model.} \\
        r_{i} & = \text{Nilai aktual untuk item ke-i, misalnya rating asli yang diberikan oleh pengguna.} \\
        \hat{r}_{i} & = \text{Prediksi untuk item ke-i yang dihasilkan oleh model.} \\
        N & = \text{Jumlah total sampel (data) yang digunakan dalam evaluasi.}
    \end{aligned}
    ```

### Perbandingan Performa Kedua Model
#### Content Based Filtering dengan Cosine Similarity
Pada metode Content Based Filtering, MSE dan RMSE digunakan untuk mengevaluasi seberapa baik sistem merekomendasikan produk berdasarkan kemiripan fitur. Dalam hal ini, kemiripan dihitung menggunakan Cosine Similarity antara produk yang diinputkan oleh pengguna dengan produk-produk yang direkomendasikan. MSE dan RMSE mengukur perbedaan antara nilai prediksi (produk yang direkomendasikan) dan nilai aktual (produk yang benar-benar dipilih atau relevan), dengan semakin kecil nilai MSE atau RMSE menunjukkan bahwa rekomendasi produk lebih akurat dan relevan dengan preferensi pengguna.

Dalam perhitungan Cosine Similarity, threshold yang digunakan adalah 0.5. Ini berarti bahwa hanya produk-produk dengan nilai similarity lebih besar dari atau sama dengan 0.5 yang akan dianggap relevan dan dimasukkan dalam daftar rekomendasi. Threshold ini dipilih untuk memastikan bahwa produk yang direkomendasikan memiliki tingkat kemiripan yang cukup signifikan dengan produk yang diinputkan pengguna, sehingga dapat meningkatkan relevansi rekomendasi. Produk dengan nilai similarity di bawah 0.5 dianggap kurang relevan dan tidak dimasukkan dalam hasil rekomendasi, untuk menjaga kualitas dan akurasi saran yang diberikan kepada pengguna.

Karena jumlah produk dalam sistem adalah 1000 produk ID, maka perhitungan MSE dan RMSE dilakukan secara rata-rata pada seluruh produk yang ada. Artinya, MSE dan RMSE dihitung untuk setiap pasangan produk yang dihitung kemiripannya menggunakan Cosine Similarity, kemudian rata-rata dari hasil-hasil tersebut diambil untuk memberikan nilai evaluasi keseluruhan terhadap akurasi rekomendasi. Dengan demikian, perhitungan ini mencerminkan performa sistem dalam merekomendasikan produk yang relevan di seluruh dataset produk yang ada.
```python
avg_mse, avg_rmse = calculate_mse_rmse_for_all(product_data, cosine_sim)
```

Hasil evaluasinya adalah sebagai berikut:
- Rata-rata MSE pada seluruh produk: 0.0711
- Rata-rata RMSE pada seluruh produk: 0.2550

Berdasarkan hasil evaluasi menggunakan MSE dan RMSE, sistem Content Based Filtering dengan menggunakan Cosine Similarity menunjukkan performa yang cukup baik dalam merekomendasikan produk yang relevan. Dengan rata-rata MSE sebesar 0.0711 dan RMSE sebesar 0.2550, sistem berhasil meminimalkan kesalahan prediksi, menunjukkan bahwa produk yang direkomendasikan memiliki tingkat relevansi yang cukup tinggi terhadap preferensi pengguna. Threshold similarity 0.5 yang diterapkan memastikan bahwa hanya produk dengan kemiripan yang signifikan yang dipertimbangkan, menjaga kualitas rekomendasi dan relevansi hasil yang diberikan.

#### Collaborative Filtering dengan Neural Collaborative Filtering (NCF)
Pada metode Collaborative Filtering dengan Neural Collaborative Filtering (NCF), MSE dan RMSE digunakan untuk mengevaluasi seberapa akurat sistem dalam memprediksi rating untuk produk yang belum pernah diberi rating oleh pengguna. Model ini mempelajari pola perilaku rating pengguna terhadap produk-produk yang telah dinilai sebelumnya dan menggunakan informasi tersebut untuk memprediksi rating produk yang belum dinilai. Dengan cara ini, NCF berusaha memberikan rekomendasi yang relevan, berdasarkan kesamaan preferensi antara pengguna dan produk yang belum diberi rating.

Berikut adalah hasil evaluasinya:
- MSE: 1.7791
- RMSE: 1.3338

MSE sebesar 1.7791 dan RMSE sebesar 1.3338 menunjukkan bahwa model memiliki akurasi yang cukup baik dalam memprediksi rating untuk produk yang belum dinilai oleh pengguna. Meskipun nilai MSE dan RMSE menunjukkan adanya ruang untuk perbaikan, model ini berhasil memanfaatkan pola perilaku rating pengguna sebelumnya untuk memberikan rekomendasi produk yang relevan. Ke depannya, improvisasi dapat dilakukan dengan memperbaiki arsitektur model, menambahkan fitur baru, atau menggunakan teknik regularisasi untuk mengurangi overfitting dan meningkatkan akurasi prediksi.

## Kesimpulan
Dalam proyek ini, dua pendekatan utama digunakan untuk membangun sistem rekomendasi produk: Content-Based Filtering dan Collaborative Filtering menggunakan Neural Collaborative Filtering (NCF). Kedua metode ini bertujuan untuk memberikan rekomendasi produk yang relevan berdasarkan preferensi pengguna, baik dengan menganalisis kemiripan fitur produk maupun dengan memanfaatkan pola perilaku rating pengguna. Evaluasi menunjukkan bahwa Content-Based Filtering dengan menggunakan Cosine Similarity berhasil memberikan rekomendasi yang sesuai dengan produk yang memiliki fitur serupa, dengan MSE dan RMSE yang menunjukkan performa yang cukup baik dalam mengukur relevansi produk berdasarkan kemiripan fitur. Sementara itu, NCF berfokus pada prediksi rating produk yang belum pernah dinilai pengguna, dengan hasil evaluasi yang memperlihatkan ruang untuk perbaikan dalam memprediksi rating akurat.

Secara keseluruhan, kedua metode menunjukkan potensi yang baik untuk meningkatkan pengalaman pengguna dalam menemukan produk yang sesuai dengan preferensi mereka. Content-Based Filtering berhasil mengidentifikasi produk serupa berdasarkan atribut yang jelas seperti kategori dan brand, sedangkan NCF memberikan rekomendasi berdasarkan preferensi yang lebih tersirat melalui interaksi pengguna sebelumnya. Meskipun keduanya memiliki hasil yang memadai, terdapat potensi untuk meningkatkan akurasi lebih lanjut melalui tuning model dan eksplorasi fitur tambahan, seperti penggabungan konteks waktu atau analisis tren pengguna.

Sebagai langkah selanjutnya, pengembangan sistem dapat berfokus pada pemanfaatan kedua pendekatan secara hybrid, dengan menggabungkan kekuatan Content-Based Filtering dan Collaborative Filtering untuk menghasilkan rekomendasi yang lebih akurat dan personal. Penerapan teknik deep learning yang lebih kompleks dan eksplorasi metode regularisasi serta fine-tuning hyperparameter akan sangat berguna untuk memperbaiki akurasi prediksi, khususnya dalam NCF, dan memberikan solusi rekomendasi yang lebih optimal di masa depan.

## Referensi
1. Ahmed MZ, Singh A, Paul A. Amazon Product Recommendation System. IJARCCE. 2022 Mar 30;11(3).
2. Dr. S. Sheeja RP. ERS: Latent Dirichlet Allocation Based E-Commerce Recommendation System Using Deep Neural Network. Psychology and Education Journal. 2021 Feb 1;58(2):953–9.
3. Rafey Ahmed Farooqi, Surabhi Kesarwani, Mohd Shakeeb, Nitin Sharma, Ishita Bhatnagar. Enhancing E-Commerce Applications with Machine Learning Recommendation Systems. International Journal of Scientific Research in Science, Engineering and Technology. 2022 May 1;85–90.
‌
‌
‌
‌
