
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


f = open("all_links.txt","a")

options = Options()
options.add_argument("--headless=new")

driver = webdriver.Chrome(options=options)

base_url = "https://indiankanoon.org/search/?formInput=commercial%20court%20%20%20doctypes%3A%20judgments"

all_links = []

for page_num in range(0, 50):

    if page_num == 0:
        url = base_url
    else:
        url = base_url + "&pagenum=" + str(page_num)

    driver.get(url)

    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'result')))

    page_source = driver.page_source

    soup = BeautifulSoup(page_source, 'html.parser')

    result_divs = soup.find_all('div', class_='result')

    for div in result_divs:
        link_element = div.find('a', href=True)
        if link_element:
            link = link_element['href']
            all_links.append(link)

driver.quit()

for link in all_links:
    link = link.split("/")
    f.write("https://indiankanoon.org/doc/"+link[2]+"\n")
