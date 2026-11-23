"""
Sistem Rekomendasi Buku (Book-Crossing) - User-based Collaborative Filtering
Tugas Data Mining and Visualization Week 10
Gregorius Sebastian Maitimo - 5003241100

Jalankan secara lokal:
    streamlit run app.py
"""

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Sistem Rekomendasi Buku", page_icon="📚", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
FILE_NAMES = {
    "users": "BX-Users.csv",
    "books": "BX-Books.csv",
    "ratings": "BX-Book-Ratings.csv",
}


# =====================================================================
# 1. DATA LOADING + EDA + CLEANING + INTEGRATION  (di-cache, hanya sekali)
# =====================================================================
def read_csv_source(src, kind):
    """Baca CSV dari bytes (upload) atau path (folder data/)."""
    buf = io.BytesIO(src) if isinstance(src, bytes) else src
    if kind == "books":
        return pd.read_csv(
            buf, sep=";", encoding="latin-1", engine="python", on_bad_lines="skip"
        )
    return pd.read_csv(buf, sep=";", encoding="latin-1")


def eda_summary(df):
    """Ringkasan struktur, missing value, dan duplikat sebuah tabel."""
    buf = io.StringIO()
    df.info(buf=buf)
    return {
        "shape": df.shape,
        "head": df.head().astype(str),
        "info": buf.getvalue(),
        "missing": pd.DataFrame(
            {
                "Jumlah Missing": df.isnull().sum(),
                "Persentase (%)": (df.isnull().mean() * 100).round(2),
            }
        ),
        "duplicates": int(df.duplicated().sum()),
    }


def normalize_isbn(s):
    return s.astype("string").str.strip().str.upper()


@st.cache_resource(
    show_spinner="Membaca, membersihkan, dan menggabungkan data "
    "(pertama kali bisa memakan waktu beberapa menit)..."
)
def prepare_data(users_src, books_src, ratings_src):
    users = read_csv_source(users_src, "users")
    books = read_csv_source(books_src, "books")
    ratings = read_csv_source(ratings_src, "ratings")

    # ---------- 2. EDA (pada data mentah) ----------
    eda = {
        "users": eda_summary(users),
        "books": eda_summary(books),
        "ratings": eda_summary(ratings),
    }
    eda["rating_counts"] = ratings["Book-Rating"].value_counts().sort_index()

    popular = ratings.groupby("ISBN").size().sort_values(ascending=False).head(10)
    eda["popular_books"] = pd.merge(
        popular.rename("Total-Ratings").reset_index(),
        books[["ISBN", "Book-Title", "Book-Author"]],
        on="ISBN",
        how="left",
    )

    # ---------- 3. DATA CLEANING ----------
    stats = {}

    # a. Missing value
    users_clean = users.copy().dropna(subset=["User-ID"])
    books_clean = books.copy().dropna(subset=["ISBN", "Book-Title"])
    ratings_clean = ratings.copy().dropna(subset=["User-ID", "ISBN", "Book-Rating"])
    stats["setelah_dropna"] = {
        "Users": users_clean.shape,
        "Books": books_clean.shape,
        "Ratings": ratings_clean.shape,
    }

    # b. Duplikat
    stats["duplikat"] = {
        "Users": int(users_clean.duplicated().sum()),
        "Books": int(books_clean.duplicated().sum()),
        "Ratings": int(ratings_clean.duplicated().sum()),
    }
    users_clean = users_clean.drop_duplicates()
    books_clean = books_clean.drop_duplicates()
    ratings_clean = ratings_clean.drop_duplicates()

    # c. Umur (nilai di luar 5-100 dianggap tidak valid)
    users_clean["Age"] = pd.to_numeric(users_clean["Age"], errors="coerce")
    users_clean.loc[
        (users_clean["Age"] < 5) | (users_clean["Age"] > 100), "Age"
    ] = float("nan")
    stats["umur_kosong"] = int(users_clean["Age"].isnull().sum())

    # d. Tahun terbit (<=0 atau > 2026 dianggap tidak valid)
    books_clean["Year-Of-Publication"] = pd.to_numeric(
        books_clean["Year-Of-Publication"], errors="coerce"
    )
    books_clean.loc[
        (books_clean["Year-Of-Publication"] <= 0)
        | (books_clean["Year-Of-Publication"] > 2026),
        "Year-Of-Publication",
    ] = float("nan")
    stats["tahun_kosong"] = int(books_clean["Year-Of-Publication"].isnull().sum())

    # e. Rating (0 = implicit feedback, dibuang untuk pemodelan)
    ratings_clean = ratings_clean[ratings_clean["Book-Rating"].between(0, 10)].copy()
    ratings_tanpanol = ratings_clean[ratings_clean["Book-Rating"] > 0].copy()
    stats["total_interaksi"] = len(ratings_clean)
    stats["rating_1_10"] = len(ratings_tanpanol)

    # f. ISBN
    books_clean["ISBN"] = normalize_isbn(books_clean["ISBN"])
    ratings_tanpanol["ISBN"] = normalize_isbn(ratings_tanpanol["ISBN"])
    del ratings_clean  # hemat memori

    # ---------- 4. DATA INTEGRATION ----------
    ratings_tanpanol["User-ID"] = pd.to_numeric(
        ratings_tanpanol["User-ID"], errors="coerce"
    )
    users_clean["User-ID"] = pd.to_numeric(users_clean["User-ID"], errors="coerce")
    books_clean = books_clean.drop_duplicates(subset=["ISBN"], keep="first")

    ratings_books = pd.merge(
        ratings_tanpanol, books_clean, on="ISBN", how="left", validate="many_to_one"
    )
    data_integrasi = pd.merge(
        ratings_books, users_clean, on="User-ID", how="left", validate="many_to_one"
    )
    stats["integrasi"] = {
        "Ratings tanpa nol": ratings_tanpanol.shape,
        "Ratings + Books": ratings_books.shape,
        "Data integrasi final": data_integrasi.shape,
    }

    return {
        "eda": eda,
        "stats": stats,
        "books_clean": books_clean,
        "data_integrasi": data_integrasi,
        "integrasi_head": data_integrasi.head().astype(str),
        "signature": data_integrasi.shape,
    }


# =====================================================================
# 5-6. FEATURE ENGINEERING + MODELING
# =====================================================================
def sparsity_info(n_users, n_books, n_ratings):
    total = n_users * n_books
    return {
        "n_users": n_users,
        "n_books": n_books,
        "n_ratings": n_ratings,
        "total_cells": total,
        "sparsity": (1 - n_ratings / total) * 100,
    }


@st.cache_resource(show_spinner="Membangun user-item matrix & model collaborative filtering...")
def build_model(
    signature, _data_integrasi, min_user, min_book, test_frac, seed, n_neighbors
):
    data_model = _data_integrasi[
        [
            "User-ID",
            "ISBN",
            "Book-Rating",
            "Book-Title",
            "Book-Author",
            "Publisher",
            "Year-Of-Publication",
        ]
    ].copy()

    # Matriks sebelum filtering (hanya statistik, tanpa membangun matriks penuh)
    sebelum = sparsity_info(
        data_model["User-ID"].nunique(), data_model["ISBN"].nunique(), len(data_model)
    )

    # Filtering user & buku aktif
    user_cnt = data_model.groupby("User-ID")["ISBN"].count()
    book_cnt = data_model.groupby("ISBN")["User-ID"].count()
    active_users = user_cnt[user_cnt >= min_user].index
    popular_books = book_cnt[book_cnt >= min_book].index

    data_filtered = data_model[
        data_model["User-ID"].isin(active_users) & data_model["ISBN"].isin(popular_books)
    ].copy()

    if data_filtered.empty:
        return None

    sesudah = sparsity_info(
        data_filtered["User-ID"].nunique(),
        data_filtered["ISBN"].nunique(),
        len(data_filtered),
    )

    # Train / test split (per user)
    test_data = data_filtered.groupby("User-ID", group_keys=False).sample(
        frac=test_frac, random_state=seed
    )
    train_data = data_filtered.drop(test_data.index)

    # Mapping & training matrix
    user_ids = np.asarray(data_filtered["User-ID"].unique(), dtype=object)
    book_ids = np.asarray(data_filtered["ISBN"].unique(), dtype=object)
    user_to_idx = {u: i for i, u in enumerate(user_ids.tolist())}
    book_to_idx = {b: i for i, b in enumerate(book_ids.tolist())}

    train_matrix = csr_matrix(
        (
            train_data["Book-Rating"].to_numpy(dtype=np.float32),
            (
                train_data["User-ID"].map(user_to_idx).to_numpy(),
                train_data["ISBN"].map(book_to_idx).to_numpy(),
            ),
        ),
        shape=(len(user_ids), len(book_ids)),
    )

    model_cf = NearestNeighbors(
        metric="cosine", algorithm="brute", n_neighbors=n_neighbors + 1, n_jobs=-1
    )
    model_cf.fit(train_matrix)

    return {
        "sebelum": sebelum,
        "sesudah": sesudah,
        "train_data": train_data,
        "test_data": test_data,
        "user_ids": user_ids,
        "book_ids": book_ids,
        "user_to_idx": user_to_idx,
        "book_to_idx": book_to_idx,
        "train_matrix": train_matrix,
        "model": model_cf,
        "n_neighbors": n_neighbors,
        "seed": seed,
        "global_mean": float(train_data["Book-Rating"].mean()),
    }


def get_neighbors(bundle, u_idx):
    """Ambil k tetangga terdekat (tanpa dirinya sendiri) beserta similarity-nya."""
    distances, indices = bundle["model"].kneighbors(
        bundle["train_matrix"][u_idx], n_neighbors=bundle["n_neighbors"] + 1
    )
    neighbors, sims = [], []
    for d, i in zip(distances[0], indices[0]):
        if i != u_idx:
            neighbors.append(i)
            sims.append(1 - d)
    return neighbors, sims


def similar_users_table(bundle, user_id):
    u_idx = bundle["user_to_idx"][user_id]
    neighbors, sims = get_neighbors(bundle, u_idx)
    return pd.DataFrame(
        {
            "User-ID": [bundle["user_ids"][i] for i in neighbors],
            "Similarity": sims,
        }
    )


# =====================================================================
# 7. EVALUASI
# =====================================================================
def evaluate(bundle, sample_n, progress=None):
    test_data = bundle["test_data"]
    if sample_n < len(test_data):
        test_data = test_data.sample(sample_n, random_state=bundle["seed"])

    train_matrix = bundle["train_matrix"]
    global_mean = bundle["global_mean"]
    actual_ratings, predicted_ratings = [], []
    neighbor_cache = {}
    fallback = 0
    total = len(test_data)

    for n, row in enumerate(test_data.itertuples(index=False), start=1):
        user, book, actual = row[0], row[1], row[2]
        u_idx = bundle["user_to_idx"][user]
        b_idx = bundle["book_to_idx"][book]

        if user not in neighbor_cache:
            neighbor_cache[user] = get_neighbors(bundle, u_idx)
        neighbors, sims = neighbor_cache[user]

        neighbor_ratings = train_matrix[neighbors, b_idx].toarray().ravel()
        mask = neighbor_ratings > 0

        if mask.sum() > 0:
            prediction = np.average(
                neighbor_ratings[mask], weights=np.array(sims)[mask]
            )
        else:
            prediction = global_mean
            fallback += 1

        actual_ratings.append(actual)
        predicted_ratings.append(prediction)

        if progress is not None and (n % 200 == 0 or n == total):
            progress.progress(n / total, text=f"Mengevaluasi {n:,} / {total:,} data uji")

    mae = mean_absolute_error(actual_ratings, predicted_ratings)
    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    return {"mae": mae, "rmse": rmse, "n": total, "fallback": fallback}


# =====================================================================
# 8. REKOMENDASI
# =====================================================================
def recommend_books(bundle, books_clean, user_id, n_recommendations=10):
    if user_id not in bundle["user_to_idx"]:
        return pd.DataFrame()

    train_matrix = bundle["train_matrix"]
    user_index = bundle["user_to_idx"][user_id]
    neighbors, sims = get_neighbors(bundle, user_index)

    neighbor_ratings = train_matrix[neighbors].toarray()
    sims = np.array(sims)

    weighted = np.dot(sims, neighbor_ratings)
    weights = np.dot(sims, (neighbor_ratings > 0).astype(int))
    predicted = np.divide(
        weighted, weights, out=np.zeros_like(weighted), where=weights != 0
    )

    # Buang buku yang sudah dinilai user
    rated = train_matrix[user_index].toarray().ravel()
    predicted[rated > 0] = 0

    top = [i for i in np.argsort(predicted)[::-1] if predicted[i] > 0][
        :n_recommendations
    ]

    rec = pd.DataFrame(
        {
            "ISBN": [bundle["book_ids"][i] for i in top],
            "Predicted-Rating": [predicted[i] for i in top],
        }
    )
    rec = pd.merge(
        rec,
        books_clean[["ISBN", "Book-Title", "Book-Author", "Publisher"]],
        on="ISBN",
        how="left",
    )
    return rec[["ISBN", "Book-Title", "Book-Author", "Publisher", "Predicted-Rating"]]


# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.header("📂 Dataset")
st.sidebar.caption(
    "Upload 3 file CSV Book-Crossing, atau letakkan di folder `data/` "
    "pada repo dengan nama file yang sama."
)
uploads = {
    key: st.sidebar.file_uploader(name, type="csv", key=f"up_{key}")
    for key, name in FILE_NAMES.items()
}


def resolve_source(key):
    if uploads[key] is not None:
        return uploads[key].getvalue()
    path = DATA_DIR / FILE_NAMES[key]
    return str(path) if path.exists() else None


sources = {k: resolve_source(k) for k in FILE_NAMES}

st.sidebar.header("⚙️ Parameter Model")
min_user = st.sidebar.number_input("Minimal rating per user", 1, 100, 5)
min_book = st.sidebar.number_input("Minimal rating per buku", 1, 100, 5)
n_neighbors = st.sidebar.number_input("Jumlah tetangga (k)", 1, 100, 10)
test_frac = st.sidebar.slider("Proporsi data uji", 0.05, 0.50, 0.20, 0.05)
seed = st.sidebar.number_input("Random state", 0, 9999, 42)

# =====================================================================
# HEADER
# =====================================================================
st.title("📚 Sistem Rekomendasi Buku")
st.caption(
    "Tugas Data Mining and Visualization Week 10 · "
    "Gregorius Sebastian Maitimo - 5003241100 · "
    "User-based Collaborative Filtering (cosine similarity) · dataset Book-Crossing"
)

missing = [FILE_NAMES[k] for k, v in sources.items() if v is None]
if missing:
    st.info(
        "Dataset belum lengkap. Upload file berikut lewat sidebar: "
        + ", ".join(f"`{m}`" for m in missing)
    )
    st.stop()

data = prepare_data(sources["users"], sources["books"], sources["ratings"])
eda, stats, books_clean = data["eda"], data["stats"], data["books_clean"]

bundle = build_model(
    data["signature"],
    data["data_integrasi"],
    int(min_user),
    int(min_book),
    float(test_frac),
    int(seed),
    int(n_neighbors),
)
if bundle is None:
    st.error(
        "Tidak ada data yang tersisa setelah filtering. "
        "Turunkan nilai minimal rating per user / buku di sidebar."
    )
    st.stop()

tabs = st.tabs(
    [
        "1 · Data Loading",
        "2 · EDA",
        "3 · Cleaning",
        "4 · Integrasi",
        "5 · Feature Engineering",
        "6 · Modeling",
        "7 · Evaluasi",
        "8 · Rekomendasi",
    ]
)

# ---------------------------------------------------------------- 1
with tabs[0]:
    st.header("1. Data Loading")
    c1, c2, c3 = st.columns(3)
    c1.metric("Users", f"{eda['users']['shape'][0]:,} × {eda['users']['shape'][1]}")
    c2.metric("Books", f"{eda['books']['shape'][0]:,} × {eda['books']['shape'][1]}")
    c3.metric("Ratings", f"{eda['ratings']['shape'][0]:,} × {eda['ratings']['shape'][1]}")
    for name in ["users", "books", "ratings"]:
        st.subheader(f"5 data pertama: {name.capitalize()}")
        st.dataframe(eda[name]["head"], use_container_width=True)

# ---------------------------------------------------------------- 2
with tabs[1]:
    st.header("2. Exploratory Data Analysis")

    st.subheader("a. Struktur dataset")
    for name in ["users", "books", "ratings"]:
        with st.expander(f"Info {name.capitalize()}"):
            st.code(eda[name]["info"])

    st.subheader("b. Missing value")
    cols = st.columns(3)
    for col, name in zip(cols, ["users", "books", "ratings"]):
        col.markdown(f"**{name.capitalize()}**")
        col.dataframe(eda[name]["missing"], use_container_width=True)

    st.subheader("c. Data duplikat")
    d1, d2, d3 = st.columns(3)
    d1.metric("Users", f"{eda['users']['duplicates']:,}")
    d2.metric("Books", f"{eda['books']['duplicates']:,}")
    d3.metric("Ratings", f"{eda['ratings']['duplicates']:,}")

    st.subheader("d. Distribusi rating")
    rating_counts = eda["rating_counts"]
    fig, ax = plt.subplots(figsize=(10, 5))
    rating_counts.plot(kind="bar", edgecolor="black", ax=ax)
    ax.set_title("Distribusi Rating Buku")
    ax.set_xlabel("Rating Buku")
    ax.set_ylabel("Jumlah")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(rating_counts.rename("Jumlah"), use_container_width=True)

    st.subheader("e. 10 buku paling populer")
    st.dataframe(eda["popular_books"], use_container_width=True)

# ---------------------------------------------------------------- 3
with tabs[2]:
    st.header("3. Data Cleaning")

    st.subheader("a. Missing value")
    st.write("Baris dengan nilai kosong pada kolom kunci dibuang.")
    st.dataframe(
        pd.DataFrame(
            {k: [f"{v[0]:,}", v[1]] for k, v in stats["setelah_dropna"].items()},
            index=["Baris", "Kolom"],
        ).T,
        use_container_width=True,
    )

    st.subheader("b. Data duplikat")
    st.dataframe(
        pd.DataFrame({"Jumlah duplikat": stats["duplikat"]}), use_container_width=True
    )

    st.subheader("c. Variabel umur")
    st.write("Umur < 5 atau > 100 dianggap tidak valid dan diubah menjadi kosong (NaN).")
    st.metric("Jumlah umur kosong", f"{stats['umur_kosong']:,}")

    st.subheader("d. Variabel tahun terbit")
    st.write("Tahun ≤ 0 atau > 2026 dianggap tidak valid dan diubah menjadi kosong (NaN).")
    st.metric("Jumlah tahun terbit kosong", f"{stats['tahun_kosong']:,}")

    st.subheader("e. Rating buku")
    st.write("Rating 0 (implicit feedback) dibuang, hanya rating 1-10 yang dipakai.")
    r1, r2 = st.columns(2)
    r1.metric("Total seluruh interaksi", f"{stats['total_interaksi']:,}")
    r2.metric("Total rating 1 - 10", f"{stats['rating_1_10']:,}")

    st.subheader("f. ISBN")
    st.write("ISBN diseragamkan: spasi dihapus dan huruf dijadikan kapital.")

# ---------------------------------------------------------------- 4
with tabs[3]:
    st.header("4. Data Integration")
    st.write("Menggabungkan tabel rating, buku, dan pengguna berdasarkan ISBN dan User-ID.")
    cols = st.columns(3)
    for col, (name, shape) in zip(cols, stats["integrasi"].items()):
        col.metric(name, f"{shape[0]:,} × {shape[1]}")
    st.dataframe(data["integrasi_head"], use_container_width=True)

# ---------------------------------------------------------------- 5
with tabs[4]:
    st.header("5. Feature Engineering")
    st.write(
        "Variabel yang dipakai: `User-ID`, `ISBN`, `Book-Rating`, `Book-Title`, "
        "`Book-Author`, `Publisher`, `Year-Of-Publication`. "
        "Data disusun menjadi matriks user-item."
    )

    sb, ss = bundle["sebelum"], bundle["sesudah"]
    st.subheader("Sebelum filtering")
    a, b, c, d = st.columns(4)
    a.metric("User", f"{sb['n_users']:,}")
    b.metric("Buku", f"{sb['n_books']:,}")
    c.metric("Rating", f"{sb['n_ratings']:,}")
    d.metric("Sparsity", f"{sb['sparsity']:.4f}%")

    st.subheader(f"Setelah filtering (user ≥ {min_user} rating, buku ≥ {min_book} rating)")
    a, b, c, d = st.columns(4)
    a.metric("User", f"{ss['n_users']:,}")
    b.metric("Buku", f"{ss['n_books']:,}")
    c.metric("Rating", f"{ss['n_ratings']:,}")
    d.metric("Sparsity", f"{ss['sparsity']:.4f}%")
    st.caption(
        f"Ukuran user-item matrix: {ss['n_users']:,} × {ss['n_books']:,} "
        f"= {ss['total_cells']:,} sel, dengan {ss['n_ratings']:,} sel berisi rating."
    )

# ---------------------------------------------------------------- 6
with tabs[5]:
    st.header("6. Modeling (Collaborative Filtering)")
    st.write(
        "User-based CF dengan **cosine similarity** (`NearestNeighbors`, brute force). "
        f"Jumlah tetangga terdekat k = {n_neighbors}."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Data training", f"{len(bundle['train_data']):,}")
    m2.metric("Data testing", f"{len(bundle['test_data']):,}")
    m3.metric("Training matrix", f"{bundle['train_matrix'].shape[0]:,} × {bundle['train_matrix'].shape[1]:,}")

    st.subheader("Contoh: pengguna yang mirip")
    ids = bundle["user_ids"].tolist()
    demo_user = st.selectbox("Pilih User-ID", ids, index=0, key="demo_user")
    sim_df = similar_users_table(bundle, demo_user)
    st.dataframe(sim_df, use_container_width=True)

# ---------------------------------------------------------------- 7
with tabs[6]:
    st.header("7. Model Evaluation")
    st.write(
        "Prediksi rating pada data uji dari rata-rata berbobot (similarity) rating tetangga. "
        "Jika tidak ada tetangga yang menilai buku tersebut, dipakai rata-rata rating global."
    )
    n_test = len(bundle["test_data"])
    sample_n = st.number_input(
        "Jumlah data uji yang dievaluasi (makin besar makin lama)",
        min_value=100,
        max_value=n_test,
        value=min(5000, n_test),
        step=500,
    )
    eval_key = (data["signature"], int(min_user), int(min_book), float(test_frac),
                int(seed), int(n_neighbors), int(sample_n))

    if st.button("▶️ Jalankan evaluasi"):
        bar = st.progress(0.0, text="Memulai evaluasi...")
        st.session_state["eval_result"] = (eval_key, evaluate(bundle, int(sample_n), bar))
        bar.empty()

    saved = st.session_state.get("eval_result")
    if saved and saved[0] == eval_key:
        res = saved[1]
        e1, e2, e3 = st.columns(3)
        e1.metric("MAE", f"{res['mae']:.4f}")
        e2.metric("RMSE", f"{res['rmse']:.4f}")
        e3.metric("Data uji dievaluasi", f"{res['n']:,}")
        st.caption(
            f"{res['fallback']:,} dari {res['n']:,} prediksi "
            f"({res['fallback'] / res['n'] * 100:.1f}%) memakai rata-rata global "
            "karena tidak ada tetangga yang menilai buku tersebut."
        )
    else:
        st.info("Klik tombol di atas untuk menghitung MAE dan RMSE.")

# ---------------------------------------------------------------- 8
with tabs[7]:
    st.header("8. Hasil Rekomendasi")
    ids = bundle["user_ids"].tolist()
    default_idx = 4638 if len(ids) > 4638 else 0
    col_a, col_b = st.columns([2, 1])
    user_id = col_a.selectbox("Pilih User-ID", ids, index=default_idx, key="rec_user")
    n_rec = col_b.slider("Jumlah rekomendasi", 1, 20, 10)

    rec = recommend_books(bundle, books_clean, user_id, n_rec)

    with st.expander("Riwayat buku yang sudah dinilai user ini (data training)"):
        hist = bundle["train_data"][bundle["train_data"]["User-ID"] == user_id]
        hist = hist[["ISBN", "Book-Title", "Book-Author", "Book-Rating"]]
        st.dataframe(
            hist.sort_values("Book-Rating", ascending=False).astype(str),
            use_container_width=True,
        )

    if rec.empty:
        st.warning("Tidak ada rekomendasi untuk user ini (tetangga tidak punya buku baru).")
    else:
        st.subheader(f"Top {len(rec)} rekomendasi untuk User-ID {user_id}")
        st.dataframe(
            rec.assign(**{"Predicted-Rating": rec["Predicted-Rating"].round(3)}),
            use_container_width=True,
        )

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(rec["Book-Title"].fillna("(judul tidak ditemukan)"), rec["Predicted-Rating"])
        ax.set_xlabel("Predicted Rating")
        ax.set_ylabel("Book Title")
        ax.set_title(f"Top {len(rec)} Book Recommendations")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
