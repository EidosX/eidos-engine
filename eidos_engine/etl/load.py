import time
import pandas as pd
import os
from tqdm import tqdm
import concurrent.futures
from datetime import datetime
import traceback
from deltalake import write_deltalake
import polars as pl
import logging
import math
import pytz
import logfire


def _setup_logging(log_file, tz):
    # Remove todos os handlers anteriores
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configura um novo logging para esta execução
    logging.basicConfig(
        filename=log_file,
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.setFormatter(TimezoneFormatter(tz,"%(asctime)s - %(levelname)s - %(message)s"))
    return logger
    
class TimezoneFormatter(logging.Formatter):
    def __init__(self, tz, fmt=None):
        super().__init__(fmt)
        self.tz = tz

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.tz)
        return dt.isoformat()
    
class ETL:
    def __init__(
            self, 
            custom_function, 
            output_folder='./elt_results', 
            output_fields_prefix='etl#',
            block_errors=0.1,
            tz = pytz.timezone('America/Sao_Paulo'),
            logfire_write_token = None,
            logfire_environment = None,
            disable_logs = False
        ):
        
        self.process_id = None
        self.logger = None
        
        self.custom_function = custom_function
    
        if not isinstance(output_fields_prefix, str):
            raise Exception("output_fields_prefix must be a string")
        self.output_fields_prefix = output_fields_prefix

        if not isinstance(output_folder, str):
            raise Exception("output_folder must be a string")
        if output_folder == "":
            raise Exception("output_folder cannot be empty")
        self.output_folder = output_folder
        
        if not isinstance(block_errors, float):
            raise Exception("block_errors must be a float")
        if block_errors < 0 or block_errors > 1:
            raise Exception("block_errors must be between 0 and 1")
        self.block_errors = block_errors

        if not isinstance(tz, pytz.tzinfo.BaseTzInfo):
            raise Exception("tz must be a pytz.timezone object")
        self.tz = tz
        
        self.logfire_write_token = logfire_write_token
        self.logfire_environment = logfire_environment
        self.disable_logs = disable_logs
        
        self.logfire_enable = False
        if self.logfire_write_token and not self.disable_logs:
            logfire.configure(console=False, token = self.logfire_write_token,environment=self.logfire_environment, scrubbing = False, service_name = 'eidos-engine')
            self.logfire_enable = True
            
    def _add_prefix_to_keys(self, d):
        return {f"{self.output_fields_prefix}{key}": value for key, value in d.items()}

    def _load_data(self, df):
        t = time.time()
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df), time.time() - t
        elif isinstance(df, pl.DataFrame):
            return df, time.time() - t
        elif isinstance(df, str):
            if not df.endswith('.parquet'):
                msg = "File must be a parquet file"
                logging.error(msg)
                raise ValueError(msg)
            if not os.path.exists(df):
                msg = f"File {df} not found"
                logging.error(msg)
                raise FileNotFoundError(msg)
            return pl.read_parquet(df), time.time() - t
    
    def _process_batch(self, batch):
        batch_final = []
        num_errors = 0
        for d in batch.iter_rows(named=True):
            try:
                t = time.time()
                resp = self.custom_function(d)
                delay = time.time() - t
            except Exception:
                num_errors += 1
                batch_final.append({**d,f'{self.output_fields_prefix}status':0, f'{self.output_fields_prefix}error': traceback.format_exc()})    
                continue
            
            if not isinstance(resp, dict):
                raise TypeError("Custom function must return dict")
            batch_final.append({**d,f'{self.output_fields_prefix}delay':delay,f'{self.output_fields_prefix}status':1,**self._add_prefix_to_keys(resp)})
    
        return batch_final, num_errors

    def _log(self, msg, log_level):
        
        msg_extra = f'{self.process_id} - {msg}'        
        
        if log_level == 'info':
            print(f'{datetime.now(self.tz).isoformat()} - INFO - {msg}')
            if not self.disable_logs:
                self.logger.info(msg_extra)
                if self.logfire_enable:
                    logfire.info(f'{datetime.now(self.tz).isoformat()} - INFO - {msg_extra}')
        elif log_level == 'error':
            print(f'{datetime.now(self.tz).isoformat()} - ERROR - {msg}')
            if not self.disable_logs:
                self.logger.error(msg_extra)
                if self.logfire_enable:
                    logfire.error(f'{datetime.now(self.tz).isoformat()} - ERROR - {msg_extra}')
        elif log_level == 'warning':
            print(f'{datetime.now(self.tz).isoformat()} - WARNING - {msg}')
            if not self.disable_logs:
                self.logger.warning(msg_extra)
                if self.logfire_enable:
                    logfire.warning(f'{datetime.now(self.tz).isoformat()} - WARNING - {msg_extra}')
        return msg

    def get_results(self, process_id, output_type):
        
        output_type = output_type.lower()
        if output_type not in ['polars', 'pandas', 'json']:
            raise Exception("output_type must be 'Polars' or 'Pandas' or 'Json'")
        
        if not os.path.exists(f"{self.output_folder}/{process_id}"):
            raise Exception(f"Process_id {process_id} not exists in output_folder")
        
        df = pl.read_delta(f"{self.output_folder}/{process_id}/result")
        if output_type == 'pandas':
            df = df.to_pandas()
        elif output_type == 'json':
            df = df.to_dicts()
            
        return df 
    
    def process_in_threads(
            self, 
            df, 
            batch_size, 
            num_threads, 
            limit = None, 
            skip = None, 
            process_id = None
            
        ):
        
        if process_id is None:
            process_id = 'eidos-engine@' + datetime.now(self.tz).isoformat()
        
        self.process_id = process_id
    
        if os.path.exists(f"{self.output_folder}/{process_id}"):
            raise Exception(f"Process_id {process_id} already exists in output_folder")
   
        os.makedirs(f"{self.output_folder}/{process_id}", exist_ok=True)    
        
        if not self.disable_logs:
            self.logger = _setup_logging(f"{self.output_folder}/{process_id}/main.log", self.tz)

        self._log(f"Starting, Process_id: {process_id}", "info")
        
        if not isinstance(df, (pd.DataFrame, pl.DataFrame, str)):
            raise Exception(self._log("df must be a pandas dataframe, polars dataframe or a string path to a parquet file", "error"))
        
        self._log("Loading data", "info")
        df, _delay = self._load_data(df)
        self._log(f"Loaded data in {_delay/60:.2f} min", "info")
 
        df = df.slice(skip or 0, limit or df.shape[0] - (skip or 0))
            
        if df.shape[0] == 0:
            raise Exception(self._log("Nothing to process", "error"))
        
        self._log("Processing", "info")
        t = time.time()
        
        for i, batch in tqdm(enumerate(df.iter_slices(n_rows=batch_size)), total=math.ceil(df.height // batch_size), desc="Processing Batches"):
            
            self._log(f"Batch {i * batch_size} -> {i * batch_size + batch_size}", "info")
            tt = time.time()
            
            height = batch.height
            _num_threads = height if num_threads > height else num_threads
            
            final_results = []
            final_num_errors = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=_num_threads) as executor:
                num = math.ceil(height / _num_threads)
                futures = [executor.submit(self._process_batch, batch.slice(j * num, num)) for j in range(_num_threads)]
                
                for batch_final, num_errors in executor.map(lambda f: f.result(), futures):
                    final_results.extend(batch_final)
                    final_num_errors += num_errors

            if final_results:
                write_deltalake(f"{self.output_folder}/{process_id}/result", pl.DataFrame(final_results), mode='append')
            if final_num_errors / height > self.block_errors:
                raise Exception(self._log(f"Too many errors in batch {i * batch_size} -> {i * batch_size + batch_size}", "error"))
            
            t_batch_time = time.time() - tt
            self._log(f"Batch {i * batch_size} -> {i * batch_size + batch_size}, {final_num_errors} errors, Finished @{t_batch_time/60:.2f} min", "info")
            self._log(f"Estimated finish time: {((math.ceil(df.height / batch_size) - (i+1)) * t_batch_time) /60:.2f} min, Throughput: {height/t_batch_time:.2f} it/s", "info")
            
        self._log(f"Processing Finished @{(time.time() - t)/60:.2f} min", "info")

        return process_id