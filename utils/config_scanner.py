class ConfigScanner:
    @staticmethod
    def print_header(title):
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)

    @staticmethod
    def select_similarity_method():
        ConfigScanner.print_header("PEMILIHAN METODE SIMILARITY")

        print("\nPilihan metode similarity yang tersedia:\n")
        print("  1. Gaussian Similarity (RBF/Kernel-based)")
        print("     - Lebih cocok untuk data yang tersebar tidak teratur")
        print("     - Menggunakan similarity berbasis kernel Gaussian")
        print("\n  2. Cosine Similarity (Vector-based)")
        print("     - Lebih cocok untuk data text/vector")
        print("     - Mengukur kesamaan sudut antara vectors")

        while True:
            try:
                choice = input("\nPilih metode (1 atau 2): ").strip()

                if choice == "1":
                    print("Gaussian Similarity dipilih")
                    return "_w_gaussian"
                elif choice == "2":
                    print("Cosine Similarity dipilih")
                    return "_w_cosine"
                else:
                    print("Input tidak valid. Pilih 1 atau 2.")

            except KeyboardInterrupt:
                print("\nInput dibatalkan oleh user")
                return "_w_gaussian"  # Default
            except Exception as e:
                print(f"Error: {e}")

    @staticmethod
    def select_number_of_clusters():
        ConfigScanner.print_header("PENENTUAN JUMLAH CLUSTER")

        print("\nMasukkan jumlah cluster yang diinginkan untuk K-Means clustering.")
        print("Rekomendasi: 2-5 clusters untuk hasil yang berarti")

        while True:
            try:
                n_clusters = input(
                    "\nBerapa jumlah cluster? (default: 2, range: 2-10): "
                ).strip()

                # Jika kosong, gunakan default
                if n_clusters == "":
                    print("Menggunakan default: 2 clusters")
                    return 2

                # Convert ke integer
                n_clusters = int(n_clusters)

                # Validasi range
                if n_clusters < 2:
                    print("Jumlah cluster minimal adalah 2")
                    continue
                elif n_clusters > 10:
                    print("Jumlah cluster maksimal adalah 10")
                    continue
                else:
                    print(f"{n_clusters} clusters dipilih")
                    return n_clusters

            except ValueError:
                print("Input harus berupa angka (integer)")
            except KeyboardInterrupt:
                print("\nInput dibatalkan oleh user")
                return 2  # Default
            except Exception as e:
                print(f"Error: {e}")

    @staticmethod
    def confirm_configuration(similarity_type=None, n_clusters=None):
        ConfigScanner.print_header("KONFIRMASI KONFIGURASI")

        print(f"\n  Metode Similarity       : {similarity_type.capitalize()}")
        print(f"  Jumlah Clusters         : {n_clusters}")
        print(f"  (K dan N_Component akan otomatis sama: {n_clusters})")

        while True:
            try:
                confirm = (
                    input("\nApakah konfigurasi sudah benar? (y/n): ").strip().lower()
                )

                if confirm in ["y", "yes"]:
                    print("\nKonfigurasi dikonfirmasi. Memulai proses...")
                    return True
                elif confirm in ["n", "no"]:
                    print("\nKonfigurasi dibatalkan. Silakan ulangi pemilihan.")
                    return False
                else:
                    print("Input tidak valid. Masukkan 'y' atau 'n'")

            except KeyboardInterrupt:
                print("\nInput dibatalkan oleh user")
                return False
            except Exception as e:
                print(f"Error: {e}")

    @staticmethod
    def scan_all_configurations():
        print("\n")
        print("KONFIGURASI SISTEM CLUSTERING INTERAKTIF".center(68))

        while True:
            similarity_type = ConfigScanner.select_similarity_method()

            # Scan number of clusters
            n_clusters = ConfigScanner.select_number_of_clusters()

            # n_components otomatis sama dengan n_clusters
            n_components = n_clusters

            # Confirm configuration
            if ConfigScanner.confirm_configuration(similarity_type, n_clusters):
                configuration = {
                    "similarity_type": similarity_type,
                    "n_clusters": n_clusters,
                    "n_components": n_components,
                }

                print("\n")
                print("  KONFIGURASI BERHASIL DIMUAT")
                print("\n")

                return configuration
            else:
                # User ingin mengulang dari awal
                continue
