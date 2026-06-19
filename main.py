from preprocessing.scraping import ReviewScraper
from preprocessing.case_folding import CaseFolding
from preprocessing.cleaning import ReviewCleaner
from preprocessing.normalization import Normalizer
from preprocessing.lemmatization import ReviewLemmatizer
from preprocessing.stopword import ReviewStopWordRemover
from preprocessing.deduplicate import ReviewDeduplicator
from preprocessing.labeling import ReviewLabeler
from clustering.split_clustering import SplitLabelClustering
from evaluation.clustering_visualization import ClusteringScatterPlotter
from evaluation.silhouette_analyzer import SilhouetteAnalyzer
from evaluation.wordcloud_runner import WordCloudRunner
from evaluation.top_words_analyzer import TopWordsAnalyzer
from utils.config_scanner import ConfigScanner
import os

product_urls = [
    # 1
    "https://www.tokopedia.com/wolfin-coid/kaos-pria-dan-wanita-ll-kaos-distro-lengan-pendek-size-m-l-xl-xxl-sablon-motif-family-berkualitas-baju-distro-lengan-pendek-1729628346453690092/review",
    # 2
    "https://www.tokopedia.com/fashion-good-t-shiet/t-shirt-rucas-ldistro-pria-wanitapremium-black-oblong-baju-polos-kaos-cowok-keren-surfing-atasan-1732245262251296123/review",
    # 3
    "https://www.tokopedia.com/hello-bello/bundle-3pcs-kaos-polos-pilih-warna-sekukamu-t-shirt-pria-lengan-pendek-kaos-oblong-1729882771725323097/review",
    # 4
    "https://www.tokopedia.com/berkat-store78/berkat-store-t-shirt-kaos-distro-katun-lengan-pendek-blue-silver-shirt-distro-premium-pria-dan-wanita-termurah-baju-kaos-str-picture-1730832397969164067/review",
    # 5
    "https://www.tokopedia.com/serba100/kaos-oversize-4-pcs-100-ribu-cowok-cewek-t-shirt-pria-wanita-distro-m-l-xl-xxl-1731590746531137386/review",
    # 6
    "https://www.tokopedia.com/dewa-muraah/paket-4pcs-100ribu-kaos-distro-bahan-semi-katun-motif-simpel-ready-size-m-l-xl-xxl-xxxl-mix-brand-1729609616854255420/review",
    # 7
    "https://www.tokopedia.com/archive-ashall-1-1695255702/ashalsky-kaos-dewasa-pria-wanita-adventure-cream-keren-unisex-baju-kaos-oblong-dewasa-lengan-pendek-kaos-distro-pria-1729700686394722041/review",
    # 8
    "https://www.tokopedia.com/albarsid/kaos-pria-baju-oversize-premium-tebal-cetaken-kartun-one-piece-luffy-vintage-longgar-kaos-washed-pria-baju-kaos-distro-oblong-pria-100-cod-kain-lembut-nyaman-panjang-pendek-kerah-t-shirt-1731151663758018226/review",
    # 9
    "https://www.tokopedia.com/ombebenglab/paket-3-pcs-kaos-polos-cotton-combed-30s-premium-baju-polos-pria-xs-paket-3-71099/review",
    # 10
    "https://www.tokopedia.com/baju-distro07/kaos-pria-dewasa-lengan-pendek-baju-katun-combed-motif-sablon-time-tries-all-the-future-is-in-sight-keren-nyaman-untuk-distro-oblong-1732432528990045475/review",
    # 11
    "https://www.tokopedia.com/rk-distro/paket-hemat-100rb-3pcs-kaos-pria-atasan-cowok-original-1729678764923390300/review",
    # 12
    "https://www.tokopedia.com/terminal-fashion-215/promo-kaos-unisex-3pcs-hanya-50k-bisa-dipakai-unisex-1729672811182851482/review",
    # 13
    "https://www.tokopedia.com/gf-grand-fashion/kaos-pria-dewasa-oblong-kaos-distro-premium-unisex-t-shirt-1731342863468627277/review",
    # 14
    "https://www.tokopedia.com/archive-ashall-1-1695255702/ashall-sky-kaos-dewasa-unisex-tshirt-distro-simpel-summer-black-kaos-pria-distro-kaos-polos-baju-motif-1729867915066182393/review",
    # 15
    "https://www.tokopedia.com/safani-shop/paket-4-pcs-kaos-polos-pocket-saku-pria-wanita-mbn-l-xl-xxl-kaos-distro-cowok-polos-saku-kantong-mbn-cewe-cowo-semi-katun-24s-baju-kaos-distro-100-ribu-dapat-4-pcs-kaos-oversize-wanita-polos-saku-jumbo-1729618358203812413/review",
]
# input dan output (default input review file)
output_review = input_review = "tokopedia_reviews.csv"

# Kamus slang
dir_dict = "custom_dict"
slang_dict_file = f"{dir_dict}/combined_slang_words.json"
root_dict_file = f"{dir_dict}/combined_root_words.txt"
stop_dict_file = f"{dir_dict}/combined_stop_words.txt"


def main() -> None:
    # Scan konfigurasi dari terminal
    config = ConfigScanner.scan_all_configurations()
    similarity = config["similarity_type"]
    n_clusters = config["n_clusters"]
    n_components = config["n_components"]

    dir = "results"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # paths untuk input/output yang bergantung pada direktori lokal
    output_casefolding = input_casefolding = f"{dir}/tokopedia_casefolding.csv"
    output_cleaned = input_cleaned = f"{dir}/tokopedia_cleaning.csv"
    output_normalized = input_normalized = f"{dir}/tokopedia_normalized.csv"
    output_lemmatized = input_lemmatized = f"{dir}/tokopedia_lemmatized.csv"
    output_stopword_removed = input_stopword_removed = (
        f"{dir}/tokopedia_stopword_removed.csv"
    )
    output_deduplicated = input_deduplicated = f"{dir}/tokopedia_deduplicated.csv"
    output_labeled = input_labeled = f"{dir}/tokopedia_labeled_all.csv"

    # Proses Scraping Reviews (pasang komentar jika file sudah ada)
    scraper = ReviewScraper()
    scraper.scrape_multiple_products(product_urls, output_review)

    # Proses Preprocessing
    caseFolded = CaseFolding(input_review, output_casefolding, "Review")
    caseFolded.run()

    cleaner = ReviewCleaner(input_casefolding, output_cleaned, "case_folded")
    cleaner.run()

    normalizer = Normalizer(
        input_cleaned, output_normalized, "cleaned_text", slang_dict_file
    )
    normalizer.run()

    lemmatizer = ReviewLemmatizer(
        input_normalized, output_lemmatized, "normalized_text", root_dict_file
    )
    lemmatizer.run()

    stopwordRemover = ReviewStopWordRemover(
        input_lemmatized, output_stopword_removed, "lemmatized_text", stop_dict_file
    )
    stopwordRemover.run()

    deduplicator = ReviewDeduplicator(
        input_stopword_removed, output_deduplicated, "stopword_text"
    )
    deduplicator.run()

    labeler = ReviewLabeler(input_deduplicated, output_labeled, "deduplicated_text")
    split_files = labeler.run()

    # Proses Clustering untuk setiap label (Split Clustering)
    print("\n" + "=" * 60)
    print("MEMULAI SPLIT LABEL CLUSTERING")
    print("=" * 60)

    split_clustering = SplitLabelClustering(
        labeled_split_dir=dir,  # Direktori results yang berisi split files
        output_dir=dir,
        similarity_type=similarity,
        n_clusters=n_clusters,
        n_components=n_components,
    )
    clustering_results = split_clustering.run()

    # Visualisasi Hasil Clustering dengan Scatter Plot
    print("\n")
    print("MEMULAI VISUALISASI CLUSTERING")

    plotter = ClusteringScatterPlotter(
        clustering_results=clustering_results,
        output_dir=dir,
        similarity_type=similarity,
        n_clusters=n_clusters,
    )
    plotter.run()

    # Analisis Silhouette Score untuk evaluasi kualitas clustering
    print("\n")
    print("MEMULAI SILHOUETTE ANALYSIS")

    silhouette_output_dir = os.path.join(dir, "silhouette_analysis")
    silhouette_analyzer = SilhouetteAnalyzer(silhouette_output_dir)

    labels = ["positif", "negatif", "all"]

    for label in labels:
        eigenvectors_file = os.path.join(
            dir, f"tokopedia_eigenvectors_{label}{similarity}.csv"
        )
        clusters_file = os.path.join(
            dir, f"tokopedia_clusters_{label}{similarity}.csv"
        )

        if os.path.exists(eigenvectors_file) and os.path.exists(clusters_file):
            silhouette_analyzer.analyze_single_label(
                eigenvectors_file=eigenvectors_file,
                clusters_file=clusters_file,
                label_name=label,
                similarity_type=similarity,
            )
        else:
            print(f"\nFile tidak ditemukan untuk label: {label}")

    # Generate dan print report
    silhouette_analyzer.generate_report()
    silhouette_analyzer.print_summary()

    # Wordcloud Visualization - Langkah terakhir evaluasi
    print("\n")
    print("MEMULAI WORDCLOUD VISUALIZATION")

    wordcloud_runner = WordCloudRunner(output_dir=dir)
    wordcloud_runner.run(similarity_type=similarity, n_clusters=n_clusters)
    print(wordcloud_runner.generate_report())

    # Top Words Analyzer - Menampilkan top 10 kata per cluster
    print("\n")
    print("MEMULAI ANALISIS TOP 10 KATA PER CLUSTER")

    top_words_analyzer = TopWordsAnalyzer(output_dir=dir)
    top_words_analyzer.analyze_top_words(similarity_type=similarity, n_clusters=n_clusters)
    print(top_words_analyzer.print_summary())

    print("\nProses selesai!")


if __name__ == "__main__":
    main()
