import time
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


class ReviewScraper:
    def __init__(self):
        self.driver = None
        self.all_data = []

    def clean_text(self, text: str) -> str:
        # Menghilangkan spasi dan enter
        return text.replace("\n", " ").strip()

    def get_review_data(self, container) -> dict[str, str]:
        # Ambil data dari container
        try:
            ulasan = (
                container.find("span", attrs={"data-testid": "lblItemUlasan"}).text
                if container.find("span", attrs={"data-testid": "lblItemUlasan"})
                else ""
            )

            if ulasan.strip():
                return {
                    "Review": self.clean_text(ulasan),
                    "Product URL": self.current_url,
                }
            return None
        except (AttributeError, TypeError):
            return None

    def load_all_reviews(self, url: str) -> list[dict]:
        self.current_url = url  # simpan URL saat ini
        if self.driver is None:
            self.driver = webdriver.Chrome()

        try:
            self.driver.get(url)
        except Exception as e:
            print(f"Error loading URL {url}: {e}")
            return []

        data = []
        page_number = 1

        while True:
            try:
                time.sleep(3)
                try:
                    buttons = self.driver.find_elements(
                        By.CSS_SELECTOR, "button.css-89c2tx"
                    )
                    for button in buttons:
                        button.click()
                        time.sleep(2)
                except Exception:
                    pass

                soup = BeautifulSoup(self.driver.page_source, "html.parser")
                review_section = soup.find("section", attrs={"id": "review-feed"})

                if review_section is None:
                    print(f"Review section not found for {url}")
                    break

                containers = review_section.find_all("article")

                for container in containers:
                    review_data = self.get_review_data(container)
                    if review_data:
                        data.append(review_data)

                print(
                    f"Product: {url} | Page {page_number} Loaded | Total reviews: {len(data)}"
                )
                page_number += 1

                try:
                    time.sleep(2)
                    next_button = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable(
                            (By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']")
                        )
                    )
                    next_button.click()
                except Exception:
                    print(f"Maximum page reached for {url}")
                    break
            except Exception as e:
                print(f"Error during scraping {url}: {e}")
                break

        return data

    def scrape_multiple_products(self, urls: list, output_file: str) -> None:
        if not urls:
            print("No URLs provided for scraping!")
            return

        # Catat waktu mulai
        start_time = time.time()
        start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("PROSES SCRAPING DIMULAI")
        print(f"Waktu mulai: {start_datetime}")

        for url in urls:
            print(f"\nStarting to scrape: {url}")
            try:
                product_reviews = self.load_all_reviews(url)
                self.all_data.extend(product_reviews)
                print(
                    f"Finished scraping {url}. Total reviews collected: {len(self.all_data)}"
                )
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue

        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass

        # Simpan hasil jika ada data
        if self.all_data:
            df = pd.DataFrame(self.all_data)
            df.to_csv(output_file, index=False)
            print(f"Scraping telah disimpan ke {output_file}")
        else:
            print("No reviews collected!")

        # Hitung dan tampilkan durasi total
        end_time = time.time()
        end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = end_time - start_time

        print(f"\nPROSES SCRAPING SELESAI")
        print(f"Waktu mulai: {start_datetime}")
        print(f"Waktu selesai: {end_datetime}")
        print(f"Total waktu: {duration:.2f} detik ({duration/60:.2f} menit)")
        print(f"Total reviews dikumpulkan: {len(self.all_data)}")
