"""Employer-name -> company mapping for ~50 large tech employers.

LCA files list legal entity names ("GOOGLE LLC", "AMAZON.COM SERVICES LLC",
"FACEBOOK INC"). We match normalized employer strings against per-company fragments
(whole-word regex). `ticker` drives the returns join; `public_since` documents how far
back the stock history is usable for the backtest.

Matching precedence matters: more specific fragments first (e.g. "meta platforms" and
"facebook" -> meta) so a generic token can't steal a row. Order in COMPANIES is preserved.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class Co:
    company_id: str
    ticker: str
    sector: str
    fragments: list[str]
    public_since: str = ""  # informational


# fragments are normalized: lowercase, alphanumerics + spaces only, matched on word boundaries
COMPANIES: list[Co] = [
    Co("nvidia","NVDA","Semis",["nvidia"],"1999"),
    Co("apple","AAPL","Hardware",["apple inc","apple computer"],"1980"),
    Co("microsoft","MSFT","Software",["microsoft"],"1986"),
    Co("alphabet","GOOGL","Internet",["google","alphabet","youtube","deepmind","waymo"],"2004"),
    Co("meta","META","Internet",["meta platforms","facebook","instagram","whatsapp","oculus"],"2012"),
    Co("amazon","AMZN","Internet",["amazon","aws","amazon web services","a2z development"],"1997"),
    Co("tesla","TSLA","Auto",["tesla"],"2010"),
    Co("netflix","NFLX","Internet",["netflix"],"2002"),
    Co("amd","AMD","Semis",["advanced micro devices","amd"],"1972"),
    Co("intel","INTC","Semis",["intel corp","intel corporation"],"1971"),
    Co("oracle","ORCL","Software",["oracle"],"1986"),
    Co("salesforce","CRM","Software",["salesforce"],"2004"),
    Co("adobe","ADBE","Software",["adobe"],"1986"),
    Co("cisco","CSCO","Networking",["cisco"],"1990"),
    Co("ibm","IBM","IT Services",["international business machines","ibm corp"],"1915"),
    Co("qualcomm","QCOM","Semis",["qualcomm"],"1991"),
    Co("broadcom","AVGO","Semis",["broadcom"],"2009"),
    Co("texas_instruments","TXN","Semis",["texas instruments"],"1953"),
    Co("micron","MU","Semis",["micron"],"1984"),
    Co("paypal","PYPL","Fintech",["paypal"],"2015"),
    Co("uber","UBER","Internet",["uber technologies","uber usa"],"2019"),
    Co("lyft","LYFT","Internet",["lyft"],"2019"),
    Co("airbnb","ABNB","Internet",["airbnb"],"2020"),
    Co("snowflake","SNOW","Software",["snowflake"],"2020"),
    Co("palantir","PLTR","Software",["palantir"],"2020"),
    Co("servicenow","NOW","Software",["servicenow"],"2012"),
    Co("workday","WDAY","Software",["workday"],"2012"),
    Co("datadog","DDOG","Software",["datadog"],"2019"),
    Co("crowdstrike","CRWD","Security",["crowdstrike"],"2019"),
    Co("zscaler","ZS","Security",["zscaler"],"2018"),
    Co("okta","OKTA","Security",["okta"],"2017"),
    Co("mongodb","MDB","Software",["mongodb"],"2017"),
    Co("twilio","TWLO","Software",["twilio"],"2016"),
    Co("snap","SNAP","Internet",["snap inc","snapchat"],"2017"),
    Co("pinterest","PINS","Internet",["pinterest"],"2019"),
    Co("roku","ROKU","Hardware",["roku"],"2017"),
    Co("block","XYZ","Fintech",["block inc","square inc"],"2015"),
    Co("shopify","SHOP","Software",["shopify"],"2015"),
    Co("atlassian","TEAM","Software",["atlassian"],"2015"),
    Co("intuit","INTU","Software",["intuit"],"1993"),
    Co("vmware","VMW","Software",["vmware"],"2007"),
    Co("dell","DELL","Hardware",["dell technologies","dell inc","dell products"],"2018"),
    Co("hp","HPQ","Hardware",["hewlett packard","hp inc","hp enterprise","hewlett-packard"],"1957"),
    Co("cloudflare","NET","Software",["cloudflare"],"2019"),
    Co("unity","U","Software",["unity technologies"],"2020"),
    Co("roblox","RBLX","Internet",["roblox"],"2021"),
    Co("doordash","DASH","Internet",["doordash"],"2020"),
    Co("coinbase","COIN","Fintech",["coinbase"],"2021"),
    Co("ebay","EBAY","Internet",["ebay"],"1998"),
    Co("applied_materials","AMAT","Semis",["applied materials"],"1972"),
]


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (name or "").lower()).strip()


class Matcher:
    """Compiled multi-company matcher: normalized employer string -> company_id or None."""
    def __init__(self, companies: list[Co] = COMPANIES):
        self.companies = companies
        self._patterns = []  # (compiled_regex, company_id), specific first
        for co in companies:
            for frag in sorted(co.fragments, key=len, reverse=True):
                self._patterns.append((re.compile(rf"\b{re.escape(frag)}\b"), co.company_id))

    def match(self, employer_name: str) -> str | None:
        s = normalize(employer_name)
        if not s:
            return None
        for pat, cid in self._patterns:
            if pat.search(s):
                return cid
        return None


def ticker_of() -> dict[str, str]:
    return {co.company_id: co.ticker for co in COMPANIES}


def all_tickers() -> list[str]:
    return [co.ticker for co in COMPANIES]
