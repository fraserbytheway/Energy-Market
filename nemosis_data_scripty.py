import pandas as pd
from nemosis import dynamic_data_compiler
from pathlib import Path

DEFAULT_START = '2025/11/01 00:00:00'
DEFAULT_END = '2025/11/25 00:00:00'
RAW_DATA_CACHE = Path.cwd() / 'raw_data'
RAW_DATA_CACHE.mkdir(parents=True, exist_ok=True)

INTS = ['int64', 'Int64']       # NumPy int64 + pandas Int64
FLOATS = ['float64', 'Float64'] # NumPy float64 + pandas Float64

# Creates table with nemosis table query format
class Table:

    # AEMO tables found here https://github.com/UNSW-CEEM/NEMOSIS/wiki/AEMO-Tables
    # filterc is column to filter on, filterv is value to select within column
    def __init__(self, name, cols, filterc = None, filterv = None):
        self.name = name
        self.cols = cols
        self.filterc = filterc
        self.filterv = filterv
        self.df = None

    # keep_csv stores raw data in cache
    def get_data(self, start = DEFAULT_START, end = DEFAULT_END, cache = RAW_DATA_CACHE, stor_bool = False):
        self.df =  dynamic_data_compiler(start, end, self.name, cache, select_columns=self.cols,
                                    filter_cols=self.filterc if self.filterc else None,
                                    filter_values=self.filterv if self.filterv else None,
                                    keep_csv=stor_bool)

    def drop_col(self, col: str):
        self.df.drop(col, axis=1, inplace=True)

    # reduce size if files get too large
    def downcast(self):
        for col in self.df.select_dtypes(include=INTS).columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='integer')

        for col in self.df.select_dtypes(include=FLOATS).columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='float')

# Creates the full bid stack offered for each market participant, with price bands
# and volume allocated to those price bands
def build_bid_stack(output_path, start = DEFAULT_START, end = DEFAULT_END):
    bdo = Table(
        name = 'BIDDAYOFFER_D',
        cols = ['DUID', 'BIDTYPE', 'DIRECTION', 'OFFERDATE', 'SETTLEMENTDATE',
            'PRICEBAND1', 'PRICEBAND2', 'PRICEBAND3', 'PRICEBAND4',
           'PRICEBAND5', 'PRICEBAND6', 'PRICEBAND7', 'PRICEBAND8', 'PRICEBAND9',
           'PRICEBAND10', 'MINIMUMLOAD', 'T1', 'T2', 'T3', 'T4'],
        filterc = ['BIDTYPE'],
        filterv = [['ENERGY']]
    )

    bpo = Table(
        name = 'BIDPEROFFER_D',
        cols = ['DUID', 'BIDTYPE', 'DIRECTION', 'INTERVAL_DATETIME',
           'OFFERDATE', 'MAXAVAIL', 'BANDAVAIL1', 'BANDAVAIL2',
           'BANDAVAIL3', 'BANDAVAIL4', 'BANDAVAIL5', 'BANDAVAIL6', 'BANDAVAIL7',
           'BANDAVAIL8', 'BANDAVAIL9', 'BANDAVAIL10', 'SETTLEMENTDATE'],
        filterc = ['BIDTYPE'],
        filterv = [['ENERGY']]
    )

    for table in [bdo, bpo]:
        table.get_data(start, end)
        table.drop_col('BIDTYPE')
        table.downcast()

    bids = bpo.df.merge(bdo.df, on = ['DUID', 'DIRECTION', 'OFFERDATE', 'SETTLEMENTDATE'], how = 'left')
    bids.drop(['OFFERDATE', 'SETTLEMENTDATE'], axis=1, inplace=True)
    bids.to_parquet(output_path)

    return bids

# Creates dataframe with actual dispatch data for each market participant,
# including region RRP for each dispatch interval
def build_dispatch_stack(output_path, datasheet_path, start = DEFAULT_START, end = DEFAULT_END):
    dl = Table(
        name = 'DISPATCHLOAD',
        cols = ['SETTLEMENTDATE', 'DUID', 'INITIALMW', 'TOTALCLEARED', 'AVAILABILITY']
    )

    dp = Table(
        name = 'DISPATCHPRICE',
        cols = ['SETTLEMENTDATE', 'REGIONID', 'RRP']
    )

    dl.get_data(start, end)
    dp.get_data(start, end)

    # need to assign region to each DUID to join to dp table
    ref = pd.read_excel(datasheet_path)
    ref = ref.loc[:, ['Category', 'Fuel Source - Primary', 'DUID', 'Region']]
    ref = ref[ref['Category'] == 'Market']
    ref.drop('Category', axis=1, inplace=True)
    ref = ref.drop_duplicates().reset_index(drop=True)
    dl.df = dl.df.merge(ref, on='DUID', how='left')
    dl.df.rename(columns={'Region': 'REGIONID'}, inplace=True)

    res = dl.df.merge(dp.df, on=['SETTLEMENTDATE', 'REGIONID'])
    res.to_parquet(output_path)

    return res

if __name__ == '__main__':

    dlp = build_dispatch_stack("dispatch.parquet", "datasheet.xlsx",
                                start = DEFAULT_START,
                                end = DEFAULT_END)
    bids = build_bid_stack("bidstack.parquet",
                           start = DEFAULT_START,
                           end = DEFAULT_END)
    if not bids.empty:
        print("---- Bid Stack Data ----")
        print("Columns: ", bids.columns, '\n\n', "Shape: ", bids.shape, '\n\n', "Head: ", bids.head)
    if not dlp.empty:
        print("---- Dispatch Stack Data ----")
        print("Columns: ", dlp.columns, '\n\n', "Shape: ", dlp.shape, '\n\n', "Head: ", dlp.head)
