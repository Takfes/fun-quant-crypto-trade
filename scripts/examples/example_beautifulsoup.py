import http.client

from bs4 import BeautifulSoup


def just_etf_info(isin):
    # request data
    conn = http.client.HTTPSConnection("www.justetf.com")
    payload = ""
    headers = {
        "Cookie": "AWSALB=Dvx3YWmWkSFbZxymUvD+co0eagkOY44AgotFowO3c9U4vlwld7DIUWJjG082AiA1BjsJ3QTaVLOUB1x3LlM7gYtNg1IAk2Xcl09b13BBV38Y0tXmrt4ywEw3qnA2; AWSALBCORS=Dvx3YWmWkSFbZxymUvD+co0eagkOY44AgotFowO3c9U4vlwld7DIUWJjG082AiA1BjsJ3QTaVLOUB1x3LlM7gYtNg1IAk2Xcl09b13BBV38Y0tXmrt4ywEw3qnA2; JSESSIONID=C2111186F3F2FB6C67E59004894B8D04; XSRF-TOKEN=251ede0c-90e7-4a8d-b8db-771133f6ad6f; locale_=en"
    }
    conn.request("GET", f"/en/etf-profile.html?isin={isin}", payload, headers)
    res = conn.getresponse()
    data = res.read()

    # parse HTML
    soup = BeautifulSoup(data, "html.parser")

    # title -----------------------------------------------------
    title = soup.find("h1", {"id": "etf-title"}).text.strip()

    # ticker -----------------------------------------------------
    ticker = soup.find("span", {"id": "etf_identifier_2"}).text.strip()

    # isin -------------------------------------------------------
    isin = soup.find("span", {"id": "etf-first-id"}).text.strip()

    # description content ----------------------------------------
    description_content = soup.find("div", {"id": "etf-description-content"}).text.strip()

    # legal structure --------------------------------------------
    table_legal_structure = soup.find("table", {"class": "table etf-data-table m-sep"})
    legal_structure = {}

    for row in table_legal_structure.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) == 2:
            key = cells[0].get_text(strip=True)
            value = cells[1].get_text(strip=True)
            legal_structure[key] = value

    ucits = legal_structure.get("UCITS compliance")

    # distribution policy ----------------------------------------
    table_data_basics = soup.find("table", {"class": "table etf-data-table"})
    data_basics = {}

    for row in table_data_basics.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) == 2:
            key = cells[0].get_text(strip=True)
            value = cells[1].get_text(strip=True)
            data_basics[key] = value

    distr_policy = data_basics.get("Distribution policy")

    # consolidate results ----------------------------------------
    result = {
        "title": title,
        "ticker": ticker,
        "isin": isin,
        "description": description_content,
        "ucits": ucits,
        "distr_policy": distr_policy,
    }

    return result


just_etf_info("IE00B3XXRP09")
