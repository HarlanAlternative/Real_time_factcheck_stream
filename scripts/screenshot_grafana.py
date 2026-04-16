"""Take a screenshot of the Grafana factcheck dashboard."""
import asyncio
from playwright.async_api import async_playwright

GRAFANA_URL = "http://localhost:3000"
DASHBOARD_URL = f"{GRAFANA_URL}/d/efj2e50yaonpca/real-time-factcheck-stream?orgId=1&from=now-1h&to=now&theme=dark&kiosk"
OUTPUT = "docs/assets/grafana-dashboard.png"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})

        # Login
        await page.goto(f"{GRAFANA_URL}/login")
        await page.fill('[name="user"]', "admin")
        await page.fill('[name="password"]', "admin")
        await page.click('[type="submit"]')
        await page.wait_for_timeout(3000)
        print(f"Logged in, current URL: {page.url}")

        # Navigate to dashboard
        await page.goto(DASHBOARD_URL)
        # Wait for panels to load
        await page.wait_for_timeout(4000)
        print(f"Navigated to dashboard: {page.url}")

        await page.screenshot(path=OUTPUT, full_page=False)
        print(f"Screenshot saved to {OUTPUT}")
        await browser.close()


asyncio.run(main())
