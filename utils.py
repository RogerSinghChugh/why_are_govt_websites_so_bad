# Class to handle and close all cookie related and other pop ups on webpages


import asyncio


class PopupHandler:
    def __init__(self):
        pass

    async def scroll_to_bottom(self, page):
        """
        Scroll to the bottom of the web page using Playwright.
        """
        print("Scrolling...")
        previous_height = await page.evaluate("document.body.scrollHeight")
        while True:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)
            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == previous_height:
                break
            previous_height = new_height
        print("Reached the bottom of the page.")    

    async def close_india_tax_identifier_popups(self, page):
        try:
            for _ in range(3):
                await page.click("text=Continue")
                await page.wait_for_timeout(2000)  # wait for 2 second between clicks
        except Exception as e:
            print(f"Continue button not found: {e}")
        # # Click on the first 'Continue' button
        # try:
        #     await page.click("text=Continue")            
        # except Exception as e:
        #     print(f"First Continue button not found: {e}")
        #     return

        # # Click on the second 'Continue' button
        # try:
        #     await page.click("text=Continue")
        # except Exception as e:
        #     print(f"Second Continue button not found: {e}")
        #     return

        # # Click on the third 'Continue' button
        # try:
        #     await page.click("text=Continue")
        # except Exception as e:
        #     print(f"Third Continue button not found: {e}")
        #     return
        # return

    async def close_uk_tax_identifier_popups(self, page):
        try:
            await page.click('text=Reject additional cookies')
            await page.wait_for_timeout(2000)  # wait for 2 seconds
            await page.click('text=Hide this message')
        except Exception as e:
            print(f"Reject additional cookies button not found: {e}")
    
    async def close_uk_tax_efiling_popups(self, page):
        await self.close_uk_tax_identifier_popups(page)
              
    async def close_uk_passport_application_popups(self, page):
        await self.close_uk_tax_identifier_popups(page)

    async def close_ireland_tax_identifier_popups(self, page):
        try:
            await page.get_by_role("button", name="Accept all Cookies").click() 
        except Exception as e:
            print(f"Accept all Cookies button not found: {e}")

    async def close_malaysia_tax_identifier_popups(self, page):
        try:
            await page.click('text=Reject All')
        except Exception as e:
            print(f"Reject All button not found: {e}")     
    
    async def close_popups(self, country: str, category: str, page):
        """
        Close pop-ups based on country and category.
        """
        if country == "India" and category == "national_tax_identifier_portal":
            await self.close_india_tax_identifier_popups(page)
        elif country == "United Kingdom" and category == "national_tax_identifier_portal":
            await self.close_uk_tax_identifier_popups(page)
        elif country == "United Kingdom" and category == "income_tax_efiling_portal":
            await self.close_uk_tax_efiling_popups(page)
        elif country == "United Kingdom" and category == "passport_application_portal":
            await self.close_uk_passport_application_popups(page)
        elif country == "Ireland" and category == "national_tax_identifier_portal":
            await self.close_ireland_tax_identifier_popups(page)
        elif country == "Malaysia" and category == "national_tax_identifier_portal":
            await self.close_malaysia_tax_identifier_popups(page)
        else:
            print(f"No specific popup handler for {country} - {category}")

        