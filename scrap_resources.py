from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import uvicorn

app = FastAPI()

chrome_options = Options()
chrome_options.add_argument('--ignore-certificate-errors')
chrome_options.add_argument('--ignore-ssl-errors')
chrome_options.add_argument('--disable-extensions')
chrome_options.add_argument('--disable-popup-blocking')

driver = webdriver.Chrome(options=chrome_options)

class InputData(BaseModel):
    input_text: str

@app.post("/scrape/")
def scrape_website(input_data: InputData):
    url = "https://plagiarismdetector.net/ar"
    input_text = input_data.input_text

    driver.get(url)

    try:
        text_area = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '//*[@id="textarea"]')))
        print("Textarea found")
        text_area.click()
        print("Inputting text into textarea")
        text_area.send_keys(input_text)

        time.sleep(30)

        button = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#check')))
        print("Button found")
        driver.execute_script("arguments[0].scrollIntoView(true);", button)
        print("Scrolled ")
        button.click()
        print("Button clicked")

        time.sleep(60)

        # Find the parent <a> tag
        parent_element = WebDriverWait(driver, 60).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, '#accordion > form > div > div.card-header.fw_600.border-0.pb-2.bg_292929.rounded-0.clr_fff.text-left > a')))
        print('element')

        # Find the <div> inside the parent <a> tag with class "side_dropdown"
        dropdown_element = WebDriverWait(driver, 60).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR,'#accordion > form > div > div.card-header.fw_600.border-0.pb-2.bg_292929.rounded-0.clr_fff.text-left > a > div')))
        print('dropdown')

        driver.execute_script("arguments[0].scrollIntoView(true);", dropdown_element)
        print("Scrolled1")

        # Click the dropdown element
        dropdown_element.click()
        print('dropdown clicked')

        result_div = WebDriverWait(driver, 60).until(EC.visibility_of_element_located((By.CSS_SELECTOR, '#collapse1 > div > a')))
        print('results found')
        resources = result_div.text

        return {"resources": resources}
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    uvicorn.run(app, host="192.168.230.75", port=8000)
