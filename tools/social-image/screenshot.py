#!/usr/bin/env python3
"""Screenshot a served social-card page at a fixed 1080px width.

Height follows the rendered content, capped at 1350px (LinkedIn's tallest
supported portrait, 4:5), so a short opening paragraph produces a shorter image
instead of one with a large trailing gap. Invoked by tools/social-image.sh once
_site_social is served locally; not meant to be run standalone.

Usage:
    python3 screenshot.py --url http://127.0.0.1:PORT/__social-card__/ --out /path/to/card.png
"""

import argparse
from playwright.sync_api import sync_playwright

WIDTH = 1080
MAX_HEIGHT = 1350


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="URL of the served social-card page")
    parser.add_argument("--out", required=True, help="Output PNG path")
    args = parser.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": WIDTH, "height": MAX_HEIGHT},
            device_scale_factor=1,
            color_scheme="light",
        )
        page.goto(args.url)
        page.wait_for_load_state("networkidle")

        # Size the viewport to the card. A full_page screenshot would not work
        # here: it never captures less than the viewport height, which is the
        # trailing gap we are trying to remove.
        height = page.evaluate(
            "Math.ceil(document.querySelector('.card').getBoundingClientRect().height)"
        )
        height = max(1, min(height, MAX_HEIGHT))
        page.set_viewport_size({"width": WIDTH, "height": height})
        page.wait_for_timeout(100)

        page.screenshot(path=args.out, full_page=False)
        browser.close()

    print(f"Wrote {args.out} ({WIDTH}x{height})")


if __name__ == "__main__":
    main()
