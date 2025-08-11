from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import UnexpectedAlertPresentException
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import requests

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import mplfinance as mpf
from sklearn.model_selection import train_test_split
import time

from tqdm import tqdm


#asynchronous libraries
# from celery import Celery
# import time

# app = Celery(
#     "scrape_dataset",broker="redis://localhost:6370/0",backend="redis://localhost:6370/0")

# dismiss the get notification part
def get_page_source_handling_alert(driver):
    try:
        return driver.page_source
    except UnexpectedAlertPresentException:
        alert = driver.switch_to.alert
        alert.dismiss()
        print("Notification popup dismissed.")
        return driver.page_source

url_name_list = [
    'ACLBSL', 'ADBL', 'AHL', 'AHPC', 'AKJCL', 'AKPL', 'ALBSL', 'ALICL', 'ANLB', 'API',
    'AVYAN', 'BARUN', 'BBC', 'BEDC', 'BFC', 'BGWT', 'BHDC', 'BHL', 'BHPL', 'BNHC',
    'BNL', 'BNT', 'BPCL', 'CBBL', 'CFCL', 'CGH', 'CHCL', 'CHDC', 'CHL', 'CIT',
    'CITY', 'CKHL', 'CLI', 'CORBL', 'CREST', 'CYCL', 'CZBIL', 'DDBL', 'DHPL', 'DLBS',
    'DOLTI', 'DORDI', 'EBL', 'EDBL', 'EHPL', 'ENL', 'FMDBL', 'FOWAD', 'GBBL', 'GBIME',
    'GBLBS', 'GCIL', 'GFCL', 'GHL', 'GILB', 'GLBSL', 'GLH', 'GMFBS', 'GMFIL', 'GMLI',
    'GRDBL', 'GUFL', 'GVL', 'HATHY', 'HBL', 'HDHPC', 'HDL', 'HEI', 'HEIP', 'HHL',
    'HIDCL', 'HIDCLP', 'HLBSL', 'HLI', 'HPPL', 'HRL', 'HURJA', 'ICFC', 'IGI', 'IHL',
    'ILBS', 'ILI', 'JBBL', 'JBLB', 'JFL', 'JOSHI', 'JSLBB', 'KBL', 'KBSH', 'KDL',
    'KKHC', 'KMCDB', 'KPCL', 'KSBBL', 'LBBL', 'LEC', 'LICN', 'LLBS', 'LSL', 'MAKAR',
    'MANDU', 'MATRI', 'MBJC', 'MBL', 'MCHL', 'MDB', 'MEHL', 'MEL', 'MEN', 'MERO',
    'MFIL', 'MHCL', 'MHL', 'MHNL', 'MKCL', 'MKHC', 'MKHL', 'MKJC', 'MLBBL', 'MLBL',
    'MLBS', 'MLBSL', 'MMKJL', 'MNBBL', 'MPFL', 'MSHL', 'MSLB', 'NABBC', 'NABIL',
    'NADEP', 'NBL', 'NESDO', 'NFS', 'NGPL', 'NHDL', 'NHPC', 'NICA', 'NICL', 'NICLBSL',
    'NIFRA', 'NIL', 'NIMB', 'NIMBPO', 'NLG', 'NLIC', 'NLICL', 'NLO', 'NMB', 'NMBMF',
    'NMFBS', 'NMIC', 'NMLBBL', 'NRIC', 'NRM', 'NRN', 'NTC', 'NUBL', 'NWCL', 'NYADI',
    'OHL', 'OMPL', 'PCBL', 'PFL', 'PHCL', 'PMHPL', 'PMLI', 'PPCL', 'PPL', 'PRIN',
    'PROFL', 'PRVU', 'PURE', 'RADHI', 'RAWA', 'RBCL', 'RFPL', 'RHGCL', 'RHPL', 'RIDI',
    'RLFL', 'RNLI', 'RSDC', 'RURU', 'SADBL', 'SAHAS', 'SALICO', 'SAMAJ', 'SANIMA',
    'SANVI', 'SAPDBL', 'SARBTM', 'SBI', 'SBL', 'SCB', 'SFCL', 'SGHC', 'SGIC', 'SHEL',
    'SHINE', 'SHIVM', 'SHL', 'SHLB', 'SHPC', 'SICL', 'SIFC', 'SIKLES', 'SINDU', 'SJCL',
    'SJLIC', 'SKBBL', 'SLBBL', 'SLBSL', 'SMATA', 'SMB', 'SMFBS', 'SMH', 'SMHL', 'SMJC',
    'SMPDA', 'SNLI', 'SONA', 'SPC', 'SPDL', 'SPHL', 'SPIL', 'SPL', 'SRLI', 'SSHL',
    'STC', 'SWBBL', 'SWMF', 'TAMOR', 'TPC', 'TRH', 'TSHL', 'TTL', 'TVCL', 'UAIL',
    'UHEWA', 'ULBSL', 'ULHC', 'UMHL', 'UMRH', 'UNHPL', 'UNL', 'UNLB', 'UPCL', 'UPPER',
    'USHEC', 'USHL', 'USLB', 'VLBS', 'VLUCL', 'WNLB','RBB', 'KRBL', 'RHPC', 'MKJC', 
    'SEF', 'SAEF', 'NICGF', 'CMF1', 'NBF2', 'CMF2',
    'NIBLSF', 'NMB50', 'SIGS2', 'NICBF', 'SFMF', 'LUK', 'NADDF', 'SLCF', 'KEF', 
    'SBCF', 'NIBSF2', 'PSF', 'NICSF', 'SSIS', 'RMF1', 'MMF1', 'NMBSBFE', 'NBF3', 
    'KDBY', 'NICFC', 'GIBF1', 'SLK', 'NSIF2', 'SAGF', 'NFCF', 'NIBLGF', 'SFEF', 
    'PRSF', 'KSLY', 'SIGS3', 'C30MF', 'RMF2', 'LVF2', 'H8020', 'NICGF2', 'NIBLSTF', 
    'KSY', 'SJLICP', 'NRICP', 'CIZBD90', 'MATRIP', 'ADLB', 'NSLB', 'KLBSL', 'SMFDB', 
    'MMFDB', 'RULB'
]

# url_name_list=['ADBL']

def extract_soup(driver,url):

    driver.get(url)

    # Scraping the first page table
    button = driver.find_element(By.ID,"ctl00_ContentPlaceHolder1_CompanyDetail1_lnkHistoryTab")
    button.click()
    time.sleep(15)
    html_content = driver.page_source
    soup = BeautifulSoup(html_content,'html.parser')
    table = soup.find('table',class_='table table-bordered table-striped table-hover')
    titles = table.find_all('th')
    table_title = [title.text.strip() for title in titles[1:]]
    df = pd.DataFrame(columns=table_title)
    return(soup,df,table_title,driver)

def scrape_table(soup,df,name,append= False):
    table = soup.find('table',class_='table table-bordered table-striped table-hover')
    try:
        column_data = table.find_all('tr')
        for row in column_data[1:]:
            row_data = row.find_all('td')
            individual_data = [data.text.strip() for data in row_data[1:]]
            # print(individual_data)
            starting_index = len(df)
            df.loc[starting_index] = individual_data

    except Exception as e:
        print("Error Occured:",{e})
    path = f'/Users/jibanchaudhary/Documents/Projects/Trading_bot/dataset/{name}.csv'
    df.to_csv(path, mode='a' if append else 'w', index=False, header=not append)



if __name__=="__main__":
    # Initializing the driver only once
    ua =  UserAgent()
    options = webdriver.ChromeOptions()
    options.add_argument(f"user-agent={ua.random}")

    ## Enabling headless will allow browser to operate in Background
    # options.add_argument("--headless") 
    prefs = {"profile.default_content_setting_values.notifications": 2}
    options.add_experimental_option("prefs", prefs)


    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    # initilization of the webDriver
    driver = webdriver.Chrome(service = Service(ChromeDriverManager().install()),options=options)

    for current_index in range(202,len(url_name_list)):
        url_name = url_name_list[current_index]
        url=f"https://merolagani.com/CompanyDetail.aspx?symbol={url_name}#0"
        name = url.split('=')[1].split('#')[0]
        soup,df,table_title,driver = extract_soup(driver,url)
        scrape_table(soup,df,name)
        # clicking the next page buttons using selenium
        for page_num in tqdm(range(2,50),desc=f"{name}"):
            try:
                driver.execute_script(f'changePageIndex("{page_num}", "ctl00_ContentPlaceHolder1_CompanyDetail1_PagerControlTransactionHistory1_hdnCurrentPage", "ctl00_ContentPlaceHolder1_CompanyDetail1_PagerControlTransactionHistory1_btnPaging")')
                time.sleep(3)
                html_content = get_page_source_handling_alert(driver)
                soup = BeautifulSoup(html_content, 'html.parser')
                table = soup.find('table',class_='table table-bordered table-striped table-hover')
                df = pd.DataFrame(columns=table_title)            
                scrape_table(soup, df, name, append=True)
            except Exception as e:
                print(f"The error occured in the page;{page_num}",{e})

    driver.quit()


# @app.task
# def main_scraper(name):
#     # Initializing the driver only once
#     ua =  UserAgent()
#     options = webdriver.ChromeOptions()
#     options.add_argument(f"user-agent={ua.random}")

#     ## Enabling headless will allow browser to operate in Background
#     # options.add_argument("--headless") 
#     prefs = {"profile.default_content_setting_values.notifications": 2}
#     options.add_experimental_option("prefs", prefs)


#     options.add_argument("--disable-gpu")
#     options.add_argument("--no-sandbox")

#     # initilization of the webDriver
#     driver = webdriver.Chrome(service = Service(ChromeDriverManager().install()),options=options)


#     url_name = name
#     url=f"https://merolagani.com/CompanyDetail.aspx?symbol={url_name}#0"
#     soup,df,table_title,driver = extract_soup(driver,url)
#     scrape_table(soup,df,name)
#     # clicking the next page buttons using selenium
#     for page_num in tqdm(range(2,50),desc=f"{name}"):
#         try:
#             driver.execute_script(f'changePageIndex("{page_num}", "ctl00_ContentPlaceHolder1_CompanyDetail1_PagerControlTransactionHistory1_hdnCurrentPage", "ctl00_ContentPlaceHolder1_CompanyDetail1_PagerControlTransactionHistory1_btnPaging")')
#             time.sleep(6)
#             html_content = get_page_source_handling_alert(driver)
#             soup = BeautifulSoup(html_content, 'html.parser')
#             table = soup.find('table',class_='table table-bordered table-striped table-hover')
#             df = pd.DataFrame(columns=table_title)            
#             scrape_table(soup, df, name, append=True)
#         except Exception as e:
#             print(f"The error occured in the page;{page_num}",{e})

#     driver.quit()