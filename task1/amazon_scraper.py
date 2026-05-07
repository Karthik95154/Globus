from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


BASE_URL = "https://www.amazon.in"
SEARCH_URL = f"{BASE_URL}/s"


def clean_text(value: str | None) -> str:
    return " ".join(value.split()) if value else ""


def get_text(node, selector: str) -> str:
    selected = node.select_one(selector)
    return clean_text(selected.get_text(" ", strip=True)) if selected else ""


def extract_price(product) -> str:
    price = get_text(product, ".a-price .a-offscreen")

    if price:
        return price

    whole = get_text(product, ".a-price-whole")
    fraction = get_text(product, ".a-price-fraction")

    if whole and fraction:
        return f"₹{whole}.{fraction}"

    if whole:
        return f"₹{whole}"

    return ""


def extract_rating(product) -> str:
    rating = get_text(product, "span.a-icon-alt")

    if rating:
        return rating.replace(" out of 5 stars", "")

    return ""


def is_ad(product) -> bool:
    sponsored_labels = [
        get_text(product, ".puis-sponsored-label-text"),
        get_text(product, ".s-sponsored-label-text"),
    ]

    return any("sponsored" in label.lower() for label in sponsored_labels)


def parse_products(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")

    rows: list[dict[str, str]] = []

    products = soup.select(
        "div.s-result-item[data-component-type='s-search-result']"
    )

    print(f"Found {len(products)} products on page")

    for product in products:
        title_node = product.select_one("h2 span")
        link_node = product.select_one("h2 a")
        image_node = product.select_one("img.s-image")

        title = clean_text(title_node.get_text()) if title_node else ""

        if not title:
            continue

        rows.append(
            {
                "image": image_node.get("src", "") if image_node else "",
                "title": title,
                "rating": extract_rating(product),
                "price": extract_price(product),
                "result_type": "Ad" if is_ad(product) else "Organic",
                "product_url": (
                    urljoin(BASE_URL, link_node.get("href", ""))
                    if link_node
                    else ""
                ),
            }
        )

    return rows


def fetch_search_page(keyword: str, page_number: int) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False
        )

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            viewport={"width": 1366, "height": 768},
            locale="en-IN",
        )

        page = context.new_page()

        url = f"{SEARCH_URL}?k={keyword}&page={page_number}"

        print(f"Opening: {url}")

        page.goto(url, timeout=60000)

        page.wait_for_timeout(
            random.randint(4000, 7000)
        )

        html = page.content()

        with open("debug.html", "w", encoding="utf-8") as f:
            f.write(html)

        browser.close()

        return html


def scrape_amazon_laptops(
    keyword: str = "laptop",
    pages: int = 1,
    delay: float = 2.0,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    seen: set[str] = set()

    for page_number in range(1, pages + 1):
        print(f"\nScraping page {page_number}")

        html = fetch_search_page(keyword, page_number)

        parsed_rows = parse_products(html)

        for row in parsed_rows:
            key = row["product_url"] or row["title"]

            if key in seen:
                continue

            seen.add(key)

            rows.append(row)

        if page_number < pages:
            sleep_time = delay + random.uniform(1, 3)

            print(f"Sleeping {sleep_time:.2f} seconds")

            time.sleep(sleep_time)

    return rows


def write_csv(
    rows: Iterable[dict[str, str]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        output_dir / f"amazon_laptops_{timestamp}.csv"
    )

    fieldnames = [
        "image",
        "title",
        "rating",
        "price",
        "result_type",
        "product_url",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape Amazon India search results "
            "into a timestamped CSV using Playwright."
        )
    )

    parser.add_argument(
        "--keyword",
        default="laptop",
        help="Search keyword",
    )

    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Number of pages to scrape",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save CSV",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between pages",
    )

    args = parser.parse_args()

    rows = scrape_amazon_laptops(
        keyword=args.keyword,
        pages=max(args.pages, 1),
        delay=max(args.delay, 0),
    )

    output_path = write_csv(
        rows,
        Path(args.output_dir),
    )

    print(f"\nSaved {len(rows)} products")
    print(f"CSV File: {output_path}")


if __name__ == "__main__":
    main()