import zipfile
import pandas as pd


class CustomCovDataLoader():
    def __init__(self, path, n_first=None):
        """
        Object for data loading. Assumes that the minutely data is stored in zip file, 
        which contains separate csv files for every stock ( each has a naming as {stock_name}_1min_data_cleaned.csv ).

        Input:
            - path : string with the direction of the zip file
        """
        # Path to the ZIP file
        self.path = path
        self.n_first = n_first

    def load_data(self):
        """
        Reads the data from a zip file from a specified location.
        
            Output : pandas dataframe with n+1 columns (timestamps and returns), where n is the number of stocks in the data.
                     1 column for timestamp and all the rest is for returns. E.g.
                     | timestamp | AAPL_pret | AMZN_pret | ... | CAT_pret |
        """

        if self.n_first is not None:
            i = 0

        # Open the ZIP file
        with zipfile.ZipFile(self.path, 'r') as zip_ref:
            # List all the files in the ZIP
            file_names = sorted(zip_ref.namelist())
            print("CSV files in the ZIP:", file_names)
            df_dict = {}
            
            merged_df = pd.DataFrame()
            for file_name in file_names:
                if file_name.endswith('.csv'):  
                    with zip_ref.open(file_name) as file:
                        df = pd.read_csv(file)
                        name = file_name.split('_1min')[0].split('/')[1]

                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df['price'] = (df['high'] + df['low'] + df['close']) / 3
                        df['pret'] = df['price'].pct_change()

                        df.rename(columns={'pret': f'{name}_pret'}, inplace=True)
                        if merged_df.empty:
                            merged_df = df[['timestamp', f'{name}_pret']]
                        else:
                            merged_df = pd.merge(merged_df, df[['timestamp', f'{name}_pret']], on='timestamp', how='outer')

                        df_dict[name] = df.copy()

                        if self.n_first is not None:
                            i += 1
                            if i >= self.n_first:
                                break


        merged_df.sort_values(by='timestamp', inplace=True)
        merged_df = merged_df.loc[1:].reset_index(drop=True) 

        return merged_df
    
    def transform_data(self, df, freq='1min'):
        """
        Transforms the data into the desired time frame frequences.
        Only 1h is supported.
        If freq is default (1min), simply sets the index of the dataframe as timestamp.

        Input:
            - freq : string indicating the desired frequency. default - 1min. The only other option is 1h.
            - merged_df : df to be modified

        Output:
            - modified merged_df
        """

        merged_df = df.copy()

        if freq not in ['1min', '1h']:
            raise ValueError(f"Invalid frequency: '{freq}'. Allowed values are '1min' or '1h'.")
        
        if freq == '1h':
            merged_df = 1 + merged_df.set_index('timestamp').copy()
            merged_df['auxiliary_col'] = 0
            cols = list(merged_df.columns)
            merged_df = merged_df.groupby(pd.Grouper(freq=freq, closed='right', label='right')).agg({col:'prod' for col in cols}) - 1
            merged_df = merged_df.loc[merged_df['auxiliary_col']==-1].copy()
            merged_df.drop('auxiliary_col', axis=1, inplace=True)

            merged_df.reset_index(inplace=True, drop=False)
            merged_df = merged_df[~merged_df['timestamp'].dt.weekday.isin([5, 6])]  # Exclude weekends
            # Exclude non-trading hours
            merged_df = merged_df[(merged_df['timestamp'].dt.time >= pd.to_datetime("10:00:00").time()) & 
                                (merged_df['timestamp'].dt.time <= pd.to_datetime("16:00:00").time())]

        merged_df.set_index('timestamp', inplace=True)

        return merged_df
