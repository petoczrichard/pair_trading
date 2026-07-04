import requests
import pandas as pd

from pair_trading.data_loader.implementations.yahoo import YahooDataSource


YAHHO_FORMAT_PARAMS = {
    "NYSE": {"exchange_code": ""},
    "NASDAQ": {"exchange_code": ""},
    "Tokyo Stock Exchange": {"exchange_code": "T"},
    "Hong Kong Exchanges And Clearing Ltd": {"exchange_code": "HK", "fill_value": "0", "width": 4},
    "National Stock Exchange Of India": {"exchange_code": "NS"},
    "London Stock Exchange": {"exchange_code": "L"},
    "Toronto Stock Exchange": {"exchange_code": "TO"},
    "Taiwan Stock Exchange": {"exchange_code": "TW", "fill_value": "0", "width": 4},
    "Korea Exchange (Stock Market)": {"exchange_code": "KS", "fill_value": "0", "width": 6},
    "Asx - All Markets": {"exchange_code": "AX"},
    "Nasdaq Omx Nordic": {"exchange_code": "ST"},
    "Xetra": {"exchange_code": "DE"},
    "Shanghai Stock Exchange": {"exchange_code": "SS", "fill_value": "0", "width": 6},
    "Nyse Euronext - Euronext Paris": {"exchange_code": "PA"},
    "SIX Swiss Exchange": {"exchange_code": "SW"},
    "Borsa Italiana": {"exchange_code": "MI"},
    "Shenzhen Stock Exchange": {"exchange_code": "SZ", "fill_value": "0", "width": 6},
    "Saudi Stock Exchange": {"exchange_code": "SR"},
    "Johannesburg Stock Exchange": {"exchange_code": "JO"},
    "Bursa Malaysia": {"exchange_code": "KL", "fill_value": "0", "width": 4},
    "Singapore Exchange": {"exchange_code": "SI"},
    "Euronext Amsterdam": {"exchange_code": "AS"},
    "Korea Exchange (Kosdaq)": {"exchange_code": "KQ", "fill_value": "0", "width": 6},
    "Bolsa De Madrid": {"exchange_code": "MC"},
    "Indonesia Stock Exchange": {"exchange_code": "JK"},
    "Tel Aviv Stock Exchange": {"exchange_code": "TA"},
    "Stock Exchange Of Thailand": {"exchange_code": "BK"},
    "Oslo Bors Asa": {"exchange_code": "OL"},
    "Omx Nordic Exchange Copenhagen A/S": {"exchange_code": "CO"},
    "Bolsa Mexicana De Valores": {"exchange_code": "MX"},
    "Gretai Securities Market": {"exchange_code": "TWO", "fill_value": "0", "width": 4},
    "Nyse Euronext - Euronext Brussels": {"exchange_code": "BB"},
    "Nasdaq Omx Helsinki Ltd.": {"exchange_code": "HE"},
    "Istanbul Stock Exchange": {"exchange_code": "IS", 'keep_root_only': True},
    "Warsaw Stock Exchange/Equities/Main Market": {"exchange_code": "WA"},
    "Santiago Stock Exchange": {"exchange_code": "SN"},
    "Nyse Mkt Llc": {"exchange_code": ""},
    "Philippine Stock Exchange Inc.": {"exchange_code": "PS"},
    "Qatar Exchange": {"exchange_code": "QA"},
    "Wiener Boerse Ag": {"exchange_code": "VI"},
    "Dubai Financial Market": {"exchange_code": "AE"},
    "New Zealand Exchange Ltd": {"exchange_code": "NZ"},
    "Abu Dhabi Securities Exchange": {"exchange_code": "AD"},
    "Athens Exchange S.A. Cash Market": {"exchange_code": "AT"},
    "Kuwait Stock Exchange": {"exchange_code": "KW"},
    "Nyse Euronext - Euronext Lisbon": {"exchange_code": "LS"},
    "Irish Stock Exchange - All Market": {"exchange_code": "IR"},
    "BSE Ltd": {"exchange_code": "BO"},
    "Budapest Stock Exchange": {"exchange_code": "BD"},
    "Egyptian Exchange": {"exchange_code": "CA"},
    "Bolsa De Valores De Colombia": {"exchange_code": "CL"},
    "Prague Stock Exchange": {"exchange_code": "PR"},
    "Cboe BZX": {"exchange_code": ""},
    "XBSP": {"exchange_code": "SA"},
}


class ISharesDataSource(YahooDataSource):

    alias = "ishares"

    def get_tickers(
        self,
        etf,
        date=None,
        only_tickers=True,
    ):
        session = requests.Session()
        headers = {"User-Agent": "Mozilla/5.0"}

        api_url = "https://www.ishares.com/us/product-screener/product-screener-v3.1.jsn"  # noqa: E501
        params = {
            "siteEntryPassthrough": "true",
            "dcrPath": "/templatedata/config/product-screener-v3/data/en/us-ishares/ishares-product-screener-backend-config",  # noqa: E501
        }

        r = session.get(api_url, params=params, headers=headers)
        r.raise_for_status()

        data = r.json()

        etf_exists = False
        for _, etf_data in data.items():
            if etf == etf_data['localExchangeTicker']:
                etf_exists = True
                break

        if not etf_exists:
            raise ValueError(f"Could find {etf} ETF.")

        date_params = (
            "&asOfDate=" + pd.Timestamp(date).strftime("%Y%m%d")
            if date is not None
            else ""
        )
        url = f"https://www.ishares.com{etf_data['productPageUrl']}/1467271812596.ajax?fileType=csv&fileName={etf_data['localExchangeTicker']}_holdings&dataType=fund" + date_params  # noqa: E501

        df = pd.read_csv(url, skiprows=9, skipfooter=2, engine="python")
        df = df[df['Asset Class'] == 'Equity']
        df = df[df['Exchange'] != 'NO MARKET (E.G. UNLISTED)']
        df = df[df['Ticker'] != '-']

        df['yahoo_ticker'] = df.apply(
            lambda row: self._yahoo_format(
                row['Ticker'],
                **YAHHO_FORMAT_PARAMS.get(row['Exchange'], {}),
            ),
            axis=1,
        )

        return df['yahoo_tickers'].tolist() if only_tickers else df

    @staticmethod
    def _yahoo_format(
        ticker,
        fill_value='',
        width=0,
        exchange_code=None,
        keep_root_only=False,
    ):
        ticker = str(ticker).upper()
        ticker = ticker.replace('*', '')
        ticker = ticker.replace('.', '-')
        ticker = ticker.replace(' ', '-')
        ticker = ticker.strip('-')

        if keep_root_only:
            ticker = ticker.split('-')[0]

        suffix = ('.' if exchange_code else '') + (exchange_code or '')
        return f'{ticker:{fill_value}>{width}}{suffix}'
