import pandas as pd
import numpy as np
from pathlib import Path
from nemosis import dynamic_data_compiler

DEFAULT_START = '2025/11/01 00:00:00'
DEFAULT_END = '2025/11/02 00:00:00'
RAW_DATA_CACHE = Path.cwd() / 'raw_data'
RAW_DATA_CACHE.mkdir(parents=True, exist_ok=True)
INTS = ['int64', 'Int64']               # numpy and pandas types
FLOATS = ['float64', 'Float64']         # numpy and pandas types
FUEL_TYPES = ['Battery Storage', 'Hydro', 'Solar', 'Fossil', 'Wind',
       'Renewable/ Biomass / Waste', '-',
       'Renewable/ Biomass / Waste and Fossil', None]


class NemRun:
    def __init__(self,
                 datasheet_path: str,
                 start: str = DEFAULT_START,
                 end: str = DEFAULT_END,
                 fuel_source: str | None = None,
                 cache: str = RAW_DATA_CACHE):
        self.datasheet_path = datasheet_path
        self.start = start
        self.end = end
        self.fuel_source = fuel_source
        self.bid_stack = pd.DataFrame()
        self.dispatch_stack = pd.DataFrame()
        self.cache = cache

        if self.fuel_source not in FUEL_TYPES:
            raise ValueError(
                f"Invalid fuel_source '{self.fuel_source}'. "
                f"Valid options: {FUEL_TYPES}"
            )

    @staticmethod
    def file_saver(df, output_path, save_parquet, save_excel):
        if save_parquet and output_path:
            df.to_parquet(output_path)

        if save_excel and output_path:
            df.to_excel(output_path)

        return

    @staticmethod
    def downcast(df):
        for col in df.select_dtypes(include=INTS).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')

        for col in df.select_dtypes(include=FLOATS).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')

        return df

    # AEMO tables found here https://github.com/UNSW-CEEM/NEMOSIS/wiki/AEMO-Tables
    # filterc is column to filter on, filterv is value to select within column
    def fetch_table(self, name, cols, filter_cols=None, filter_values=None):
        return dynamic_data_compiler(self.start, self.end, name, self.cache,
                                        select_columns=cols,
                                        filter_cols=filter_cols,
                                        filter_values=filter_values,
                                        keep_csv=False)

    # constructs full bid stack with offering and price bands
    def build_bid_stack(self, save_parquet=False, save_excel=False, output_path=None):
        bdo = self.fetch_table(
            name='BIDDAYOFFER_D',
            cols=['DUID', 'BIDTYPE', 'DIRECTION', 'OFFERDATE', 'SETTLEMENTDATE',
                  'PRICEBAND1', 'PRICEBAND2', 'PRICEBAND3', 'PRICEBAND4',
                  'PRICEBAND5', 'PRICEBAND6', 'PRICEBAND7', 'PRICEBAND8', 'PRICEBAND9',
                  'PRICEBAND10', 'MINIMUMLOAD', 'T1', 'T2', 'T3', 'T4'],
            filter_cols=['BIDTYPE'],
            filter_values=[['ENERGY']]
        )

        bpo = self.fetch_table(
            name='BIDPEROFFER_D',
            cols=['DUID', 'BIDTYPE', 'DIRECTION', 'INTERVAL_DATETIME',
                  'OFFERDATE', 'MAXAVAIL', 'BANDAVAIL1', 'BANDAVAIL2',
                  'BANDAVAIL3', 'BANDAVAIL4', 'BANDAVAIL5', 'BANDAVAIL6', 'BANDAVAIL7',
                  'BANDAVAIL8', 'BANDAVAIL9', 'BANDAVAIL10', 'SETTLEMENTDATE'],
            filter_cols=['BIDTYPE'],
            filter_values=[['ENERGY']]
        )

        for table in [bdo, bpo]:
            print(table.columns)
            table = table.drop('BIDTYPE', axis = 1)
            table = self.downcast(table)

        bids = bpo.merge(bdo, on=['DUID', 'DIRECTION', 'OFFERDATE', 'SETTLEMENTDATE'], how='left')
        bids.drop(['OFFERDATE', 'SETTLEMENTDATE'], axis=1, inplace=True)

        # add fuel source and region labels
        df = pd.read_excel(self.datasheet_path).loc[:, ['DUID', 'Fuel Source - Primary', 'Region']].rename(
            columns={'Region': 'REGIONID'})
        bids = bids.merge(df, on=['DUID'])
        bids = bids.drop_duplicates()

        self.file_saver(bids, output_path, save_parquet, save_excel)
        self.bid_stack = bids

        return bids

    # constructs full dispatch stack with volume cleared and regional reference prices
    def build_dispatch_stack(self, save_parquet=False, save_excel=False, output_path=None):
        dl = self.fetch_table(
            name='DISPATCHLOAD',
            cols=['SETTLEMENTDATE', 'DUID', 'INITIALMW', 'TOTALCLEARED', 'AVAILABILITY']
        )

        dp = self.fetch_table(
            name='DISPATCHPRICE',
            cols=['SETTLEMENTDATE', 'REGIONID', 'RRP']
        )

        # need to assign region to each DUID to join to dp table
        ref = pd.read_excel(self.datasheet_path)
        ref = ref.loc[:, ['Category', 'Fuel Source - Primary', 'DUID', 'Region']]
        ref = ref[ref['Category'] == 'Market']
        ref.drop('Category', axis=1, inplace=True)
        ref = ref.drop_duplicates().reset_index(drop=True)
        dl = dl.merge(ref, on='DUID', how='left')
        dl.rename(columns={'Region': 'REGIONID'}, inplace=True)

        res = dl.merge(dp, on=['SETTLEMENTDATE', 'REGIONID'])

        self.file_saver(res, output_path, save_parquet, save_excel)
        self.dispatch_stack = res

        return res

    # full idx from start to end date, 5 minute increments referenced to the bid stack
    # as only one price needs to be offered for the day, fills in missing prices
    def complete_bid_stack(self, save_parquet=False, save_excel=False, output_path=None):
        df = self.bid_stack if not self.bid_stack.empty else self.build_bid_stack()

        if self.fuel_source:
            df = df[df['Fuel Source - Primary'] == self.fuel_source]

        idx_start = df['INTERVAL_DATETIME'].min()
        idx_end = df['INTERVAL_DATETIME'].max()

        full_idx = pd.date_range(idx_start, idx_end, freq='5min')

        df = df.sort_values(['DUID', 'DIRECTION', 'INTERVAL_DATETIME'])
        out = []
        for (duid, direction), g in df.groupby(['DUID', 'DIRECTION'], sort=False):
            g = g.set_index('INTERVAL_DATETIME').reindex(full_idx)
            g.index.name = 'INTERVAL_DATETIME'
            g['DUID'] = duid
            g['DIRECTION'] = direction
            out.append(g.reset_index())

        df = pd.concat(out, ignore_index=False)

        # Filling in blank price bands as each interval automatically uses previous dispatch windows price band
        df = df.sort_values(['DUID', 'DIRECTION', 'INTERVAL_DATETIME'])
        to_fill = [
            'PRICEBAND1', 'PRICEBAND2', 'PRICEBAND3', 'PRICEBAND4', 'PRICEBAND5', 'PRICEBAND6',
            'PRICEBAND7', 'PRICEBAND8', 'PRICEBAND9', 'PRICEBAND10', 'MINIMUMLOAD',
            'T1', 'T2', 'T3', 'T4'
        ]

        df[to_fill] = df.groupby(['DUID', 'DIRECTION'], sort=False)[to_fill].ffill()

        self.file_saver(df, output_path, save_parquet, save_excel)

        return df

    # subset of the full bid stack, containing only bids that were dispatched
    def dispatched_bid_stack(self, save_parquet=False, save_excel=False, output_path=None):
        source = self.complete_bid_stack()
        df = self.dispatch_stack if not self.dispatch_stack.empty else self.build_dispatch_stack()
        df = df[df['TOTALCLEARED'] != 0].copy()
        df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])

        conditions = [df['TOTALCLEARED'] < 0, df['TOTALCLEARED'] == 0, df['TOTALCLEARED'] > 0]
        choices = ['LOAD', 'non', 'GEN']
        df['dispatch'] = np.select(conditions, choices, default='non')

        if self.fuel_source:
            df = df[df['Fuel Source - Primary'] == self.fuel_source]

        df = df.loc[:, ['DUID', 'SETTLEMENTDATE', 'REGIONID', 'dispatch', 'TOTALCLEARED', 'RRP']]
        df = source.merge(df, left_on=['DUID', 'REGIONID', 'DIRECTION', 'INTERVAL_DATETIME'],
                          right_on=['DUID', 'REGIONID', 'dispatch', 'SETTLEMENTDATE'], how='inner')

        df = df.drop(['Fuel Source - Primary', 'SETTLEMENTDATE', 'dispatch'], axis=1)

        self.file_saver(df, output_path, save_parquet, save_excel)

        return df

    # subset of the dispatched bid stack, containing only the specific bid bands within the dispatched bid
    # that were activated
    def activated_bids(self, save_parquet=False, save_excel=False, output_path=None):
        
        def gen_band_finder(row):
            target = row['TOTALCLEARED']
            total = 0
            for i in range(1, 11):
                total += row[f'BANDAVAIL{i}']

                if total >= target:
                    break
            if i < 10:
                for i in range(i + 1, 11):
                    row[f'BANDAVAIL{i}'] = np.nan
            return row

        def load_band_finder(row):
            target = abs(row['TOTALCLEARED'])
            total = 0
            for i in range(10, 0, -1):
                total += row[f'BANDAVAIL{i}']

                if total >= target:
                    break
                    
            if i > 1:
                for i in range(i - 1, 0, -1):
                    row[f'BANDAVAIL{i}'] = np.nan

            return row
        
        df = self.dispatched_bid_stack()
        df_gen = df[df['DIRECTION'] == 'GEN'].copy()
        df_gen = df_gen.apply(gen_band_finder, axis=1)

        df_load = df[df['DIRECTION'] == 'LOAD'].copy()
        df_load = df_load.apply(load_band_finder, axis=1)

        df = pd.concat([df_gen, df_load])
        
        self.file_saver(df, output_path, save_parquet, save_excel)
        
        return df

    # changes bid stack to have a unique price/vol band for each DUID and time interval on each row
    def bid_stack_melt(self, df = pd.DataFrame(), save_parquet=False, save_excel=False, output_path=None, discard_empty=True):
        if df.empty:
            df = self.activated_bids()

        price_cols = [c for c in df.columns if c.startswith('PRICEBAND')]
        price_melt = df.melt(
            id_vars=['DUID', 'DIRECTION', 'INTERVAL_DATETIME', 'MAXAVAIL', 'REGIONID'],
            value_vars=price_cols,
            var_name="band",
            value_name="band_price"
        )

        size_cols_prefix = "BANDAVAIL"
        size_cols = [c for c in df.columns if c.startswith(size_cols_prefix)]

        size_melt = df.melt(
            id_vars=['DUID', 'DIRECTION', 'INTERVAL_DATETIME', 'REGIONID'],
            value_vars=size_cols,
            var_name="band",
            value_name="band_vol"
        )

        price_melt['bandno'] = price_melt['band'].str.extract(r'(\d+)$').astype('int')
        size_melt['bandno'] = size_melt['band'].str.extract(r'(\d+)$').astype('int')

        df = price_melt.merge(
            size_melt,
            on=['DUID', 'DIRECTION', 'INTERVAL_DATETIME', 'REGIONID', 'bandno'],
            how='inner'
        )

        if discard_empty:
            df = df[df['band_vol'] != 0]

        self.file_saver(df, output_path, save_parquet, save_excel)

        return df

if __name__ == '__main__':

    query = NemRun(datasheet_path='datasheet.xlsx', fuel_source='Battery Storage')
    df = query.bid_stack_melt()
    print(df.head(5))

