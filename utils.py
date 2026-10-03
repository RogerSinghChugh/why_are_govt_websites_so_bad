"""Pop-up and cookie-banner handling for government portal screenshots.

Strategy per page:
  1. run any country/category-specific steps (e.g. India TIN "Continue" interstitials)
  2. run a generic pass that clicks the first visible reject / accept / close button
     matching a multilingual list of accessible names
  3. as a fallback, hide well-known consent-manager containers with CSS
"""

import asyncio
import re

from playwright.async_api import Page

# Tried in order. "Reject" variants first so fewer trackers load, then "accept", then plain dismiss.
# Each pattern is matched case-insensitively against the accessible name of buttons and links.
GENERIC_BUTTON_PATTERNS = [
    # reject / decline  (en, pt, id/ms, fr, ja)
    r"^(reject|decline|refuse)( all)?( additional| optional| non-essential)?( cookies)?$",
    r"^(rejeitar|recusar)( todos)?( os)?( cookies)?$",
    r"^tolak( semua)?$",
    r"^(refuser|tout refuser)( les cookies)?$",
    r"^(拒否する?|すべて拒否)$",
    # accept / agree
    r"^(accept|allow|agree( and close)?|i agree|ok(ay)?|got it|i understand|understood)( all)?( cookies)?$",
    r"^(aceitar|concordo|entendi)( todos)?( os)?( cookies)?$",
    r"^(terima|setuju|saya setuju|mengerti)( semua)?$",
    r"^(accepter|tout accepter|j'accepte)( les cookies)?$",
    r"^(同意する?|すべて許可|許可する)$",
    # dismiss
    r"^(close|dismiss|hide (this |cookie )?message|no thanks)$",
    r"^(fechar|tutup|fermer|閉じる)$",
    r"^(x|×|✕)$",  # bare close glyph, e.g. passports.govt.nz cookie strip
]

# Containers injected by common consent managers. Hidden as a last resort if clicking failed.
CONSENT_CONTAINER_SELECTORS = [
    "#onetrust-banner-sdk", "#onetrust-consent-sdk", ".onetrust-pc-dark-filter",
    "#CybotCookiebotDialog", "#CybotCookiebotDialogBodyUnderlay",
    ".cky-consent-container", ".cky-overlay",
    "#cookie-law-info-bar", ".cc-window", ".cc-banner",
    "#usercentrics-root", ".qc-cmp2-container", "#truste-consent-track",
    "#cookieConsent", "#cookie-consent", "#cookie-banner", "#cookieBanner",
    ".cookie-banner", ".cookie-consent", ".cookie-notice", "#cookie-notice",
    "#cookiescript_injected", "#cookies-eu-banner", ".cookiealert",
]


class PopupHandler:
    def __init__(self, max_generic_clicks: int = 3, click_timeout_ms: int = 2000, verbose: bool = True):
        self.max_generic_clicks = max_generic_clicks
        self.click_timeout_ms = click_timeout_ms
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------ generic
    async def _try_click(self, page: Page, pattern: str) -> bool:
        rx = re.compile(pattern, re.I)
        for role in ("button", "link"):
            loc = page.get_by_role(role, name=rx)
            try:
                n = await loc.count()
            except Exception:
                continue
            for i in range(min(n, 3)):
                el = loc.nth(i)
                try:
                    if await el.is_visible():
                        text = (await el.inner_text()).strip()[:40]
                        await el.click(timeout=self.click_timeout_ms)
                        await page.wait_for_timeout(600)
                        self._log(f"    clicked {role} '{text}'")
                        return True
                except Exception:
                    continue
        return False

    async def close_generic(self, page: Page) -> int:
        """Click visible banner buttons in priority order. Returns number of clicks made."""
        clicks = 0
        for pattern in GENERIC_BUTTON_PATTERNS:
            if clicks >= self.max_generic_clicks:
                break
            if await self._try_click(page, pattern):
                clicks += 1
        return clicks

    async def hide_consent_containers(self, page: Page):
        css = ", ".join(CONSENT_CONTAINER_SELECTORS) + " { display: none !important; visibility: hidden !important; }"
        try:
            await page.add_style_tag(content=css)
        except Exception as e:
            self._log(f"    could not inject consent-hiding css: {e}")

    # ---------------------------------------------------------------- specific
    async def close_india_tax_identifier_popups(self, page: Page):
        # tinpan.proteantech.in opens a chain of notice modals, each with its own "Continue" button.
        rx = re.compile(r"^\s*continue\s*$", re.I)
        for _ in range(6):
            btn = None
            for role in ("button", "link"):
                loc = page.get_by_role(role, name=rx)
                try:
                    if await loc.count() and await loc.first.is_visible():
                        btn = loc.first
                        break
                except Exception:
                    continue
            if btn is None:
                # a follow-up modal may still be animating in
                await page.wait_for_timeout(1500)
                try:
                    loc = page.get_by_role("button", name=rx)
                    if not (await loc.count() and await loc.first.is_visible()):
                        break
                    btn = loc.first
                except Exception:
                    break
            try:
                await btn.click(timeout=3000)
                self._log("    clicked India 'Continue' modal")
                await page.wait_for_timeout(1500)
            except Exception:
                break

    async def close_uk_popups(self, page: Page):
        # gov.uk: reject the cookie banner, then dismiss the "You have rejected additional cookies" bar
        for name in (re.compile(r"^reject additional cookies$", re.I), re.compile(r"^hide (this|cookie) message$", re.I)):
            try:
                btn = page.get_by_role("button", name=name).first
                if await btn.is_visible():
                    await btn.click(timeout=self.click_timeout_ms)
                    await page.wait_for_timeout(500)
            except Exception:
                pass

    SPECIFIC = {
        ("India", "national_tax_identifier_portal"): "close_india_tax_identifier_popups",
        ("United Kingdom", "national_tax_identifier_portal"): "close_uk_popups",
        ("United Kingdom", "income_tax_efiling_portal"): "close_uk_popups",
        ("United Kingdom", "passport_application_portal"): "close_uk_popups",
    }

    # ------------------------------------------------------------------ entry
    async def close_popups(self, country: str, category: str, page: Page):
        """Specific steps first, then the generic pass, then CSS fallback."""
        handler_name = self.SPECIFIC.get((country, category))
        if handler_name:
            self._log(f"    specific handler: {handler_name}")
            await getattr(self, handler_name)(page)
        n = await self.close_generic(page)
        self._log(f"    generic pass: {n} click(s)")
        await self.hide_consent_containers(page)

    # ---------------------------------------------------------------- helpers
    async def scroll_to_bottom(self, page: Page):
        """Scroll to the bottom of the page so lazy content loads (not used for viewport shots)."""
        previous_height = await page.evaluate("document.body.scrollHeight")
        while True:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)
            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == previous_height:
                break
            previous_height = new_height
        await page.evaluate("window.scrollTo(0, 0)")
