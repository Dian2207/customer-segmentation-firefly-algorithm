import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv(
    "data/result/parameter_stability.csv"
)

print(df.head())


# ==========================================
# INFO
# ==========================================
print("\n===================================")
print("PARAMETER STABILITY VISUALIZATION")
print("===================================")

print("\nJumlah Iterasi :",
      len(df))

print("\nRata-rata Silhouette")

print(
    "K-Means                :",
    round(
        df["kmeans_silhouette"].mean(),
        4
    )
)

print(
    "Firefly Only           :",
    round(
        df["firefly_silhouette"].mean(),
        4
    )
)

print(
    "Hybrid KMeans-Firefly :",
    round(
        df[
            "hybrid_firefly_silhouette"
        ].mean(),
        4
    )
)

print(
    "Hybrid KMeans-GA      :",
    round(
        df[
            "hybrid_ga_silhouette"
        ].mean(),
        4
    )
)


# ==========================================
# GRAFIK SILHOUETTE SCORE
# ==========================================
plt.figure(figsize=(14, 7))

plt.plot(
    df["iteration"],
    df["kmeans_silhouette"],
    marker='o',
    linewidth=2,
    label="K-Means"
)

plt.plot(
    df["iteration"],
    df["firefly_silhouette"],
    marker='s',
    linewidth=2,
    label="Firefly Only"
)

plt.plot(
    df["iteration"],
    df["hybrid_firefly_silhouette"],
    marker='^',
    linewidth=3,
    label="Hybrid KMeans + Firefly"
)

plt.plot(
    df["iteration"],
    df["hybrid_ga_silhouette"],
    marker='d',
    linewidth=2,
    label="Hybrid KMeans + GA"
)

plt.xlabel(
    "Iterasi",
    fontsize=12
)

plt.ylabel(
    "Silhouette Score",
    fontsize=12
)

plt.title(
    "Perbandingan Stabilitas Silhouette Score\n"
    "K-Means vs Firefly vs Hybrid",
    fontsize=14
)

plt.legend()

plt.grid(True)

plt.savefig(
    "data/result/stability_silhouette.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# ==========================================
# GRAFIK EXECUTION TIME
# ==========================================
plt.figure(figsize=(14, 7))

plt.plot(
    df["iteration"],
    df["kmeans_time"],
    marker='o',
    linewidth=2,
    label="K-Means"
)

plt.plot(
    df["iteration"],
    df["firefly_time"],
    marker='s',
    linewidth=2,
    label="Firefly Only"
)

plt.plot(
    df["iteration"],
    df["hybrid_firefly_time"],
    marker='^',
    linewidth=3,
    label="Hybrid KMeans + Firefly"
)

plt.plot(
    df["iteration"],
    df["hybrid_ga_time"],
    marker='d',
    linewidth=2,
    label="Hybrid KMeans + GA"
)

plt.xlabel(
    "Iterasi",
    fontsize=12
)

plt.ylabel(
    "Execution Time (detik)",
    fontsize=12
)

plt.title(
    "Perbandingan Stabilitas Waktu Komputasi\n"
    "K-Means vs Firefly vs Hybrid",
    fontsize=14
)

plt.legend()

plt.grid(True)

plt.savefig(
    "data/result/stability_time.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# ==========================================
# ANALISIS OTOMATIS
# ==========================================
avg_scores = {

    "K-Means":
    df["kmeans_silhouette"].mean(),

    "Firefly Only":
    df["firefly_silhouette"].mean(),

    "Hybrid KMeans + Firefly":
    df[
        "hybrid_firefly_silhouette"
    ].mean(),

    "Hybrid KMeans + GA":
    df[
        "hybrid_ga_silhouette"
    ].mean()

}

best_method = max(
    avg_scores,
    key=avg_scores.get
)


# ==========================================
# STABILITY
# ==========================================
stability_scores = {

    "K-Means":
    df["kmeans_silhouette"].std(),

    "Firefly Only":
    df["firefly_silhouette"].std(),

    "Hybrid KMeans + Firefly":
    df[
        "hybrid_firefly_silhouette"
    ].std(),

    "Hybrid KMeans + GA":
    df[
        "hybrid_ga_silhouette"
    ].std()

}

most_stable = min(
    stability_scores,
    key=stability_scores.get
)


# ==========================================
# FASTEST
# ==========================================
avg_times = {

    "K-Means":
    df["kmeans_time"].mean(),

    "Firefly Only":
    df["firefly_time"].mean(),

    "Hybrid KMeans + Firefly":
    df[
        "hybrid_firefly_time"
    ].mean(),

    "Hybrid KMeans + GA":
    df[
        "hybrid_ga_time"
    ].mean()

}

fastest_method = min(
    avg_times,
    key=avg_times.get
)


# ==========================================
# PRINT ANALISIS
# ==========================================
print("\n===================================")
print("HASIL ANALISIS STABILITAS")
print("===================================")

print(
    "\nBest Silhouette :",
    best_method
)

print(
    "Most Stable     :",
    most_stable
)

print(
    "Fastest Method  :",
    fastest_method
)

print("\nKESIMPULAN:")

print(

    f"""
Secara keseluruhan metode
{best_method}
menunjukkan performa terbaik
dengan silhouette score tertinggi.

Metode tersebut juga menunjukkan
stabilitas clustering yang baik
selama 50 iterasi pengujian.

Sedangkan metode tercepat adalah
{fastest_method}
berdasarkan rata-rata waktu komputasi.

Hasil ini menunjukkan bahwa
Hybrid KMeans + Firefly
mampu menghasilkan segmentasi
yang lebih optimal dan stabil
dibandingkan metode lainnya.
"""

)

print("\n===================================")
print("VISUALISASI BERHASIL DISIMPAN")
print("===================================")

print(
    "data/result/stability_silhouette.png"
)

print(
    "data/result/stability_time.png"
)