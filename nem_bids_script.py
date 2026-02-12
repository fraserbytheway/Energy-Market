import warnings
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from nemosis import dynamic_data_compiler

DEFAULT_START = '2025/12/01 00:00:00'
DEFAULT_END = '2025/12/31 00:00:00'
RAW_DATA_CACHE = Path.cwd() / 'raw_data'
RAW_DATA_CACHE.mkdir(parents=True, exist_ok=True)
INTS = ['int64', 'Int64']  # numpy and pandas types
FLOATS = ['float64', 'Float64']  # numpy and pandas types
FUEL_TYPES = ['Battery Storage', 'Hydro', 'Solar', 'Fossil', 'Wind',
              'Renewable/ Biomass / Waste', '-',
              'Renewable/ Biomass / Waste and Fossil', None]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class NemRun:

    """
    A class to manage the fetching, processing and analysis of AEMO data
    * there is a ~ 1 month lag of which time intervals can be accessed, due to AEMO archiving data timeframe

    Attributes:
        datasheet_path: Path to reference Excel sheet, which contains DUID info
        start: Start date for fetching
        end: End date for fetching
        fuel_source: Fuel type to filter by
        cache: Directory for storing raw AEMO CSV's
    """

    def __init__(self,
                 datasheet_path: str,
                 start: str = DEFAULT_START,
                 end: str = DEFAULT_END,
                 fuel_source: str | None = None,
                 cache: str = RAW_DATA_CACHE):
        self.datasheet_path = datasheet_path
        self.start = start
        self.end = end
        self.bid_stack = pd.DataFrame()
        self.dispatch_stack = pd.DataFrame()
        self.cache = cache

        if fuel_source not in FUEL_TYPES:
            raise ValueError(
                f"Invalid fuel_source '{fuel_source}'. "
                f"Valid options: {FUEL_TYPES}"
            )
        self.fuel_source = fuel_source


    @staticmethod
    def file_saver(df, output_path, save_parquet, save_excel):
        """
        Helper method to save Dataframes to a disc

        Args:
            df: The DataFrame to save.
            output_path: The full path (including filename/extension base) to save to.
            save_parquet: Boolean flag to save as parquet.
            save_excel: Boolean flag to save as excel.
        """
        if not output_path or df.empty:
            return

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if save_parquet:
            logger.info(f"Saving Parquet to {path.with_suffix('.parquet')}")
            df.to_parquet(path.with_suffix('.parquet'))

        if save_excel:
            logger.info(f"Saving Excel to {path.with_suffix('.xlsx')}")
            df.to_excel(path.with_suffix('.xlsx'))

    @staticmethod
    def downcast(df):
        """Reduce memory usage by downcasting numeric types"""
        fcols = df.select_dtypes('float').columns
        icols = df.select_dtypes('integer').columns

        df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
        df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

        return df

    def fetch_table(self, name, cols, filter_cols=None, filter_values=None):
        """
        Wrapper for nemosis dynamic_data_compiler to fetch AEMO data.
        AEMO tables found here https://github.com/UNSW-CEEM/NEMOSIS/wiki/AEMO-Tables
        Args:
            filter_cols: List of columns to filter by
            filter_values: Value to select within each column
        """
        logger.info(f"Fetching table: {name}...")

        return dynamic_data_compiler(self.start, self.end, name, self.cache,
                                     select_columns=cols,
                                     filter_cols=filter_cols,
                                     filter_values=filter_values,
                                     keep_csv=True)

    def build_bid_stack(self, save_parquet=False, save_excel=False, output_path=None):
        """Construct the master bid stack with offer prices and bands"""
        logger.info("Building Bid Stack...")

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
            table = table.drop('BIDTYPE', axis=1)
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
        """Constructs dispatch data with volume cleared and regional prices."""
        logger.info("Building Dispatch Stack...")
        dl = self.fetch_table(
            name='DISPATCHLOAD',
            cols=['SETTLEMENTDATE', 'DUID', 'INITIALMW', 'TOTALCLEARED', 'AVAILABILITY']
        )

        cols_to_numeric = ['TOTALCLEARED', 'INITIALMW', 'AVAILABILITY']
        for col in cols_to_numeric:
            if col in dl.columns:
                dl[col] = pd.to_numeric(dl[col], errors='coerce').fillna(0)

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

        dl['REGIONID'] = dl['REGIONID'].astype(str)
        dp['REGIONID'] = dp['REGIONID'].astype(str)
        dl['SETTLEMENTDATE'] = pd.to_datetime(dl['SETTLEMENTDATE'])
        dp['SETTLEMENTDATE'] = pd.to_datetime(dp['SETTLEMENTDATE'])

        res = dl.merge(dp, on=['SETTLEMENTDATE', 'REGIONID'])

        self.file_saver(res, output_path, save_parquet, save_excel)
        self.dispatch_stack = res

        return res

    def complete_bid_stack(self, save_parquet=False, save_excel=False, output_path=None):
        """
        Resamples the bid stack to 5-minute intervals.
        BidDayOffer only provides daily prices, so we forward-fill them to match 5-min intervals.
        """
        logger.info("Resampling Bid Stack to 5-min intervals...")
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

    def dispatched_bid_stack(self, save_parquet=False, save_excel=False, output_path=None):
        """Filters the bid stack to only include rows where dispatch actually occurred."""
        logger.info("Merging dispatch data with bid stack...")
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

    def bid_hunter(self, df=pd.DataFrame(), save_parquet=False, save_excel=False, output_path=None, col='TOTALCLEARED'):
        """
        Logic to determine exactly which bid bands were activated to meet the Total Cleared target.
        Use col='TOTALCLEARED' for dispatched bids, col='MAXAVAIL' for effective bids
        Calculates the marginal band.
        """
        logger.info("Calculating marginal bands...")

        if df.empty:
            df = self.dispatched_bid_stack()

        target_vol = df[col].abs()
        gen_idx = df['DIRECTION'] == 'GEN'
        load_idx = df['DIRECTION'] == 'LOAD'

        df_gen = df[gen_idx].copy()
        remaining = target_vol[gen_idx]

        for i in range(1, 11):
            taken = np.minimum(df_gen[f'BANDAVAIL{i}'], remaining)
            df_gen[f'BANDAVAIL{i}'] = taken
            remaining = np.maximum(remaining - taken, 0)

        df_load = df[load_idx].copy()
        remaining = target_vol[load_idx]

        for i in range(10, 0, -1):
            taken = np.minimum(df_load[f'BANDAVAIL{i}'], remaining)
            df_load[f'BANDAVAIL{i}'] = taken
            remaining = np.maximum(remaining - taken, 0)

        df = pd.concat([df_gen, df_load])

        self.file_saver(df, output_path, save_parquet, save_excel)

        return df

    def bid_stack_melt(self, df=pd.DataFrame(), save_parquet=False, save_excel=False, output_path=None,
                       discard_empty=True):
        """Reshapes the bid stack from wide to long format. Useful for agg and data analysis"""
        logger.info("Converting wide bid stack to long format...")
        if df.empty:
            df = self.bid_hunter()

        id_vars = ['DUID', 'DIRECTION', 'INTERVAL_DATETIME', 'REGIONID', 'MAXAVAIL']

        df_long = pd.wide_to_long(
            df,
            stubnames=['PRICEBAND', 'BANDAVAIL'],
            i=id_vars,
            j='bandno'
        ).reset_index()

        df_long = df_long.rename(columns={
            'PRICEBAND': 'band_price',
            'BANDAVAIL': 'band_vol'
        })

        if discard_empty:
            df_long = df_long[(df_long['band_vol'] != 0) & (df_long['band_vol'].notnull())]

        self.file_saver(df_long, output_path, save_parquet, save_excel)

        return df_long


if __name__ == '__main__':
    query = NemRun(datasheet_path='raw_data/datasheet.xlsx', fuel_source='Battery Storage')
    df = query.bid_stack_melt()
    print(df.head(5))
