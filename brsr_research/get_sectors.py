import pandas as pd
import numpy as np

OUTPUT_FOLDER = r"./output"

# Load our stock list
df = pd.read_parquet(f"{OUTPUT_FOLDER}/liquidity.parquet")
symbols = df["SYMBOL"].unique()
print(f"Total unique symbols: {len(symbols)}")

# NSE sector mapping — top stocks by sector
# Source: NSE India sectoral indices composition
sector_map = {
    # Banking & Finance
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "AXISBANK": "Banking", "KOTAKBANK": "Banking", "INDUSINDBK": "Banking",
    "BANDHANBNK": "Banking", "FEDERALBNK": "Banking", "IDFCFIRSTB": "Banking",
    "PNB": "Banking", "BANKBARODA": "Banking", "CANBK": "Banking",
    "UNIONBANK": "Banking", "INDIANB": "Banking", "MAHABANK": "Banking",
    "RBLBANK": "Banking", "DCBBANK": "Banking", "KARNATAKA": "Banking",
    "SOUTHBANK": "Banking", "J&KBANK": "Banking", "LAKSHVILAS": "Banking",
    # NBFC & Insurance
    "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC", "CHOLAFIN": "NBFC",
    "MUTHOOTFIN": "NBFC", "MANAPPURAM": "NBFC", "SUNDARMFIN": "NBFC",
    "IIFL": "NBFC", "M&MFIN": "NBFC", "SHRIRAMFIN": "NBFC",
    "HDFCLIFE": "Insurance", "SBILIFE": "Insurance", "ICICIPRULI": "Insurance",
    "LICI": "Insurance", "NIACL": "Insurance", "GICRE": "Insurance",
    # IT & Technology
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "TECHM": "IT", "LTIM": "IT", "MPHASIS": "IT", "COFORGE": "IT",
    "PERSISTENT": "IT", "HEXAWARE": "IT", "NIITTECH": "IT",
    "OFSS": "IT", "KPITTECH": "IT", "TATAELXSI": "IT", "BIRLASOFT": "IT",
    "MASTEK": "IT", "SONATSOFTW": "IT", "RAMSARUP": "IT",
    # Oil & Gas / Energy
    "RELIANCE": "Energy", "ONGC": "Energy", "IOC": "Energy",
    "BPCL": "Energy", "HPCL": "Energy", "GAIL": "Energy",
    "PETRONET": "Energy", "OIL": "Energy", "MRPL": "Energy",
    "HINDPETRO": "Energy", "AEGASIND": "Energy", "GSPL": "Energy",
    "IGL": "Energy", "MGL": "Energy", "ATGL": "Energy",
    "ADANIGREEN": "Energy", "TATAPOWER": "Energy", "NTPC": "Energy",
    "POWERGRID": "Energy", "COALINDIA": "Energy", "NHPC": "Energy",
    "SJVN": "Energy", "TORNTPOWER": "Energy", "CESC": "Energy",
    # Pharma & Healthcare
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "BIOCON": "Pharma", "LUPIN": "Pharma",
    "AUROPHARMA": "Pharma", "TORNTPHARM": "Pharma", "ALKEM": "Pharma",
    "IPCALAB": "Pharma", "GLENMARK": "Pharma", "NATCOPHARM": "Pharma",
    "PFIZER": "Pharma", "ABBOTINDIA": "Pharma", "SANOFI": "Pharma",
    "APOLLOHOSP": "Healthcare", "MAXHEALTH": "Healthcare", "FORTIS": "Healthcare",
    "METROPOLIS": "Healthcare", "THYROCARE": "Healthcare",
    # Auto & Auto Ancillaries
    "TATAMOTORS": "Auto", "MARUTI": "Auto", "M&M": "Auto",
    "BAJAJ-AUTO": "Auto", "HEROMOTOCO": "Auto", "EICHERMOT": "Auto",
    "ASHOKLEY": "Auto", "TVSMOTOR": "Auto", "FORCE": "Auto",
    "BOSCHLTD": "AutoAncil", "MOTHERSON": "AutoAncil", "BHARATFORG": "AutoAncil",
    "BALKRISIND": "AutoAncil", "MRF": "AutoAncil", "APOLLOTYRE": "AutoAncil",
    "EXIDEIND": "AutoAncil", "AMARAJABAT": "AutoAncil", "SUNDRMFAST": "AutoAncil",
    # Metals & Mining
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "VEDL": "Metals", "SAIL": "Metals", "NMDC": "Metals",
    "HINDCOPPER": "Metals", "NATIONALUM": "Metals", "MOIL": "Metals",
    "APLAPOLLO": "Metals", "WELSPUNIND": "Metals", "JINDALSAW": "Metals",
    # FMCG & Consumer
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG", "MARICO": "FMCG",
    "GODREJCP": "FMCG", "COLPAL": "FMCG", "EMAMILTD": "FMCG",
    "TATACONSUM": "FMCG", "VBL": "FMCG", "UBL": "FMCG",
    "RADICO": "FMCG", "MCDOWELL-N": "FMCG",
    # Cement & Construction
    "ULTRACEMCO": "Cement", "SHREECEM": "Cement", "AMBUJACEM": "Cement",
    "ACC": "Cement", "DALBHARAT": "Cement", "RAMCOCEM": "Cement",
    "JKCEMENT": "Cement", "HEIDELBERG": "Cement", "BIRLACORPN": "Cement",
    "LT": "Construction", "NCC": "Construction", "KNR": "Construction",
    "PNCINFRA": "Construction", "IRCON": "Construction", "RVNL": "Construction",
    # Telecom & Media
    "BHARTIARTL": "Telecom", "IDEA": "Telecom", "TATACOMM": "Telecom",
    "HFCL": "Telecom", "RAILTEL": "Telecom",
    "ZEEL": "Media", "SUNTV": "Media", "PVRINOX": "Media",
    # Real Estate
    "DLF": "RealEstate", "GODREJPROP": "RealEstate", "OBEROIRLTY": "RealEstate",
    "PRESTIGE": "RealEstate", "PHOENIXLTD": "RealEstate", "SOBHA": "RealEstate",
    "BRIGADE": "RealEstate", "MAHLIFE": "RealEstate",
    # Capital Goods & Industrials
    "SIEMENS": "CapGoods", "ABB": "CapGoods", "HAVELLS": "CapGoods",
    "CROMPTON": "CapGoods", "VOLTAS": "CapGoods", "BHEL": "CapGoods",
    "BEL": "CapGoods", "HAL": "CapGoods", "COCHINSHIP": "CapGoods",
    "GRINDWELL": "CapGoods", "CUMMINSIND": "CapGoods", "THERMAX": "CapGoods",
    # Chemicals & Specialty
    "PIDILITIND": "Chemicals", "ASIANPAINT": "Chemicals", "BERGER": "Chemicals",
    "KANSAINER": "Chemicals", "ATUL": "Chemicals", "NAVINFLUOR": "Chemicals",
    "DEEPAKNTR": "Chemicals", "FINOLEXIND": "Chemicals", "VINATIORGA": "Chemicals",
    # Consumer Durables
    "TITAN": "ConsumerDurables", "WHIRLPOOL": "ConsumerDurables",
    "BLUESTAR": "ConsumerDurables", "SYMPHONY": "ConsumerDurables",
    "VGUARD": "ConsumerDurables", "ORIENTELEC": "ConsumerDurables",
    # Logistics & Transport
    "CONCOR": "Logistics", "BLUEDART": "Logistics", "GATI": "Logistics",
    "MAHINDCIE": "Logistics", "ADANIPORTS": "Logistics",
    # Agriculture
    "UPL": "Agri", "COROMANDEL": "Agri", "RALLIS": "Agri",
    "BAYER": "Agri", "PI": "Agri", "KAVERI": "Agri",
}

# Build sector dataframe
sector_df = pd.DataFrame([
    {"SYMBOL": k, "SECTOR": v} for k, v in sector_map.items()
])

# For unmapped symbols assign "Other"
all_symbols_df = pd.DataFrame({"SYMBOL": symbols})
sector_df = all_symbols_df.merge(sector_df, on="SYMBOL", how="left")
sector_df["SECTOR"] = sector_df["SECTOR"].fillna("Other")

# Coverage stats
mapped = (sector_df["SECTOR"] != "Other").sum()
print(f"Mapped to named sector : {mapped}")
print(f"Assigned to Other      : {len(sector_df) - mapped}")
print(f"\nSector distribution:")
print(sector_df["SECTOR"].value_counts())

sector_df.to_csv(f"{OUTPUT_FOLDER}/sector_map.csv", index=False)
print(f"\nSaved sector_map.csv")