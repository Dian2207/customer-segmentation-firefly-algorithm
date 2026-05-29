import pandas as pd
import numpy as np

from comparison.evaluate_kmeans_comparison import (
    evaluate_kmeans
)

from comparison.evaluate_firefly_comparison import (
    evaluate_hybrid_firefly,
    evaluate_firefly_only
)

from comparison.evaluate_ga_comparison import (
    evaluate_ga
)


# ==========================================
# RESULT CONTAINER
# ==========================================
results = []


# ==========================================
# ITERASI 50X
# ==========================================
for iteration in range(1, 51):

    print("\n================================")
    print(f"ITERASI {iteration}/50")
    print("================================")

    np.random.seed(None)

    result = {

        "iteration": iteration

    }


    # ======================================
    # K-MEANS
    # ======================================
    print("\n[1] K-MEANS")

    kmeans_result = evaluate_kmeans()

    result["kmeans_silhouette"] = (
        kmeans_result["silhouette"]
    )

    result["kmeans_time"] = (
        kmeans_result["time"]
    )


    # ======================================
    # FIREFLY
    # ======================================
    print("\n[2] FIREFLY")

    firefly_result = (
        evaluate_firefly_only()
    )

    result["firefly_silhouette"] = (
        firefly_result["silhouette"]
    )

    result["firefly_time"] = (
        firefly_result["time"]
    )


    # ======================================
    # HYBRID K-MEANS + GA
    # ======================================
    print("\n[3] HYBRID K-MEANS + GA")

    ga_result = evaluate_ga()

    result["hybrid_ga_silhouette"] = (
        ga_result["silhouette"]
    )

    result["hybrid_ga_time"] = (
        ga_result["time"]
    )


    # ======================================
    # HYBRID K-MEANS + FIREFLY
    # ======================================
    print("\n[4] HYBRID K-MEANS + FIREFLY")

    hybrid_firefly_result = (
        evaluate_hybrid_firefly()
    )

    result[
        "hybrid_firefly_silhouette"
    ] = (
        hybrid_firefly_result["silhouette"]
    )

    result[
        "hybrid_firefly_time"
    ] = (
        hybrid_firefly_result["time"]
    )


    # ======================================
    # SAVE ITERATION RESULT
    # ======================================
    results.append(result)

    print("\nHASIL ITERASI")

    print(
        f"K-Means                    : "
        f"{result['kmeans_silhouette']:.4f}"
    )

    print(
        f"Firefly                    : "
        f"{result['firefly_silhouette']:.4f}"
    )

    print(
        f"Hybrid K-Means + GA        : "
        f"{result['hybrid_ga_silhouette']:.4f}"
    )

    print(
        f"Hybrid K-Means + Firefly   : "
        f"{result['hybrid_firefly_silhouette']:.4f}"
    )


# ==========================================
# DATAFRAME
# ==========================================
df = pd.DataFrame(results)


# ==========================================
# SAVE CSV
# ==========================================
df.to_csv(

    "data/result/parameter_stability.csv",

    index=False

)


# ==========================================
# SUMMARY
# ==========================================
print("\n================================")
print("PARAMETER STABILITY SUMMARY")
print("================================")

print("\nRATA-RATA SILHOUETTE")

print(
    "K-Means                    :",
    round(
        df["kmeans_silhouette"].mean(),
        4
    )
)

print(
    "Firefly                    :",
    round(
        df["firefly_silhouette"].mean(),
        4
    )
)

print(
    "Hybrid K-Means + GA        :",
    round(
        df[
            "hybrid_ga_silhouette"
        ].mean(),
        4
    )
)

print(
    "Hybrid K-Means + Firefly   :",
    round(
        df[
            "hybrid_firefly_silhouette"
        ].mean(),
        4
    )
)


# ==========================================
# EXECUTION TIME
# ==========================================
print("\nRATA-RATA EXECUTION TIME")

print(
    "K-Means                    :",
    round(
        df["kmeans_time"].mean(),
        4
    )
)

print(
    "Firefly                    :",
    round(
        df["firefly_time"].mean(),
        4
    )
)

print(
    "Hybrid K-Means + GA        :",
    round(
        df[
            "hybrid_ga_time"
        ].mean(),
        4
    )
)

print(
    "Hybrid K-Means + Firefly   :",
    round(
        df[
            "hybrid_firefly_time"
        ].mean(),
        4
    )
)


# ==========================================
# STANDARD DEVIATION
# ==========================================
print("\nSTANDARD DEVIATION")

print(
    "K-Means                    :",
    round(
        df["kmeans_silhouette"].std(),
        4
    )
)

print(
    "Firefly                    :",
    round(
        df["firefly_silhouette"].std(),
        4
    )
)

print(
    "Hybrid K-Means + GA        :",
    round(
        df[
            "hybrid_ga_silhouette"
        ].std(),
        4
    )
)

print(
    "Hybrid K-Means + Firefly   :",
    round(
        df[
            "hybrid_firefly_silhouette"
        ].std(),
        4
    )
)


# ==========================================
# BEST METHOD
# ==========================================
avg_scores = {

    "K-Means":
    df["kmeans_silhouette"].mean(),

    "Firefly":
    df["firefly_silhouette"].mean(),

    "Hybrid K-Means + GA":
    df[
        "hybrid_ga_silhouette"
    ].mean(),

    "Hybrid K-Means + Firefly":
    df[
        "hybrid_firefly_silhouette"
    ].mean()

}


best_method = max(
    avg_scores,
    key=avg_scores.get
)


# ==========================================
# MOST STABLE
# ==========================================
stability_scores = {

    "K-Means":
    df["kmeans_silhouette"].std(),

    "Firefly":
    df["firefly_silhouette"].std(),

    "Hybrid K-Means + GA":
    df[
        "hybrid_ga_silhouette"
    ].std(),

    "Hybrid K-Means + Firefly":
    df[
        "hybrid_firefly_silhouette"
    ].std()

}


most_stable = min(
    stability_scores,
    key=stability_scores.get
)


# ==========================================
# FASTEST METHOD
# ==========================================
avg_times = {

    "K-Means":
    df["kmeans_time"].mean(),

    "Firefly":
    df["firefly_time"].mean(),

    "Hybrid K-Means + GA":
    df[
        "hybrid_ga_time"
    ].mean(),

    "Hybrid K-Means + Firefly":
    df[
        "hybrid_firefly_time"
    ].mean()

}


fastest_method = min(
    avg_times,
    key=avg_times.get
)


# ==========================================
# FINAL ANALYSIS
# ==========================================
print("\n================================")
print("HASIL ANALISIS STABILITAS")
print("================================")

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

print(f"""

Berdasarkan pengujian sebanyak
50 iterasi, metode
{best_method}
memiliki rata-rata silhouette
score terbaik.

Metode yang paling stabil adalah
{most_stable}
berdasarkan standard deviation
terkecil.

Sedangkan metode tercepat adalah
{fastest_method}
berdasarkan rata-rata execution
time terkecil.

Hasil ini menunjukkan bahwa
Hybrid K-Means + Firefly mampu
memberikan performa clustering
yang optimal dan stabil dibanding
metode lainnya.

""")


# ==========================================
# FILE INFO
# ==========================================
print("\n================================")
print("FILE BERHASIL DISIMPAN")
print("================================")

print(
    "data/result/parameter_stability.csv"
)