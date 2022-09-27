#%%
import pandas as pd
import numpy as np
import datetime
from yahoofinancials import YahooFinancials
from scipy.optimize import minimize

def OptFun(initAllct,HistNP):
    Alloc = np.append(initAllct,(1-sum(initAllct)))
    if np.max(Alloc) > 1 or np.min(Alloc) < 0:
        Ess = np.Inf
    else:
        Prtf = np.nansum(HistNP*Alloc[None,:],1)
        Vol = np.std(Prtf)
        Rtn = np.mean(Prtf)
        Ess = Vol - Rtn
    return Ess

ticklist = ['ABC','ABMD','BAX','CVS','DGX','HZNP','HCA','UNH','JNJ','ABBV', \
    'NVO','MDT', 'AZN', 'AMGN', 'GILD', 'SYK']
NmStks = len(ticklist)
Output = pd.DataFrame()

# Constants
enddate = datetime.date.today().strftime("%Y-%m-%d")
x = range(2010, 2022, 1)
for Year in x:
    start_date = str(Year)+'-01-01'
    end_date = enddate
    time_interval = 'daily'

    # load in benchmark as the S&P 500
    Bnch = YahooFinancials('^GSPC')
    BnchRaw = Bnch.get_historical_price_data(start_date,end_date,time_interval)
    BnchRaw = pd.DataFrame(BnchRaw['^GSPC']['prices'])
    BnchRaw = BnchRaw.set_index(BnchRaw.formatted_date,drop=True)
    BnchRaw.index.names = ['Date']

    HistPrice = pd.DataFrame(data=BnchRaw.adjclose)
    HistPrice.columns = ['SP500']
    #%%
    # load in the target stock data
    for ticker in ticklist:
        myprtfl = YahooFinancials(ticker)
        WkPrcRaw = myprtfl.get_historical_price_data(start_date, end_date, time_interval)
        WkPrcRaw = pd.DataFrame(WkPrcRaw[ticker]['prices'])
        WkPrcRaw = WkPrcRaw.set_index(WkPrcRaw.formatted_date,drop=True)
        WkPrcRaw.index.names = ['Date']
        StkPrice = pd.DataFrame(data=WkPrcRaw.adjclose)
        StkPrice.columns = [ticker]
        HistPrice = pd.merge(HistPrice,StkPrice, on='Date',how='left')
    
    HistPricePrcnt = HistPrice.pct_change(axis = 'rows').iloc[1:,:]
    HistPricePrcntExBen = HistPricePrcnt.drop(columns = 'SP500')
    HistNP = HistPricePrcntExBen.to_numpy()
    initAllct = np.repeat(1/NmStks, NmStks -1)
    # Ess = OptFun(initAllct,HistNP)
    OutAllctn = minimize(OptFun, initAllct,args=(HistNP), method='Nelder-Mead', \
        options={'gtol': 1e-6, 'disp': False})
    OptAllctn = np.append(OutAllctn.x,1-sum(OutAllctn.x))
    OptAllctn = pd.DataFrame(OptAllctn[None,:],columns=[ticklist],index=[Year])
    Output = Output.append(OptAllctn)


Output.to_csv('AllocationByYear.csv')
# %%
