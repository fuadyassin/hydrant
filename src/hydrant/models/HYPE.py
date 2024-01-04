# load the packages



import pandas as pd
import numpy as np
from typing import Dict, Union
import geopandas as gpd
import sys
from itertools import product
import re


def GeoClass (combination,
              outfile,
              mapping = {'SLC':'ID','landcover':'LULC','soil':'SOIL TYPE'},
              commented_lines = '! HYPE GeoClass'):
    
    # rename the dataframe
    flipped_mapping = {v: k for k, v in mapping.items()}
        
    combination.rename(columns=flipped_mapping, inplace=True)
    
    print(combination)
    
    # order the first two 
    combination = combination[['SLC','landcover','soil']]
    
    # populate the extra rows
    combination['Main crop cropid'] = 0
    combination['Second crop cropid'] = 0
    combination['Crop rotation group'] = 0
    combination['Vegetation type'] = 1
    combination['Special class code'] = 0
    combination['Tile depth'] = 0
    combination['Stream depth'] = 2.296
    combination['Number of soil layers'] = 3
    combination['Soil layer depth 1'] = 0.091
    combination['Soil layer depth 2'] = 0.493
    combination['Soil layer depth 3'] = 2.296
    
    # write the file
    # Open the file in write mode
    with open(outfile, 'w') as file:
        # Write the commented lines
        #for line in commented_lines:
        file.write(commented_lines + '\n')

    # writing the `GeoClass.txt` file
    with open(outfile, 'a') as file:
            combination.to_csv(file, sep='\t', index=False, header=False)


def GaoData (*dfs: pd.DataFrame,
             df_mappings: Union[Dict[str, Dict[str, str]], None] = None,
             outfile):
    
    # turn input into list
    dfs = list(dfs)
    
    # Loop through each DataFrame
    for idx, df in enumerate(dfs, start=1):
        
        # get the mapping for a given df
        mapping = df_mappings.get(f'df{idx}', {})
        ID = mapping.get('id', None)
        rename_dict = mapping.get('rename_dict', {})
        
        
        # Set the ID column to index for the DataFrame
        if ID is not None:
            df.index = df[ID].astype(int)
            df.drop(columns=[ID], inplace=True)
            df = df.sort_index()
            
        # rename the df and keep renamed columns
        if rename_dict:
            df = rename_columns (df,
                                 rename_dict = rename_dict)
        
        # update the dataframe
        dfs[idx - 1] = df
    
    # Compare the indexes
    index_comparison = all(dfs[i].index.equals(dfs[i + 1].index) for i in range(len(dfs) - 1))
    if index_comparison:
        print("The indexes of all DataFrames are exactly the same with the same order.")
    else:
        sys.exit("The indexes of DataFrames are different or in a different order.")
        
    # merge all the dfs horizontally
    merged_df  = pd.concat(dfs,axis=1)
    
    # order based on the upstream area (from upstream to downstream)
    merged_df = merged_df.sort_values(by='up_area').reset_index(drop=True)
    merged_df = merged_df.drop(columns=['up_area'])
    merged_df.to_csv(outfile, sep='\t', index=False)

def rename_columns (df,
                    rename_dict: Dict[str, str] = None,
                    keep_renamed_only = True):
    
    # initialize column name list
    col_name_list = []
    
    # loop over the key and values of the rename_dictionary
    for key, value in rename_dict.items():
        
        # find the level of columns that start from the given key
        filtered_cols = sorted(col for col in df.columns if col.startswith(value))
        
        if len(filtered_cols) > 1: # multiple columns start with given key

            # loop over each col name and rename it based on 
            for filtered_col in filtered_cols:
            
                # Extract the numeric part from the string
                numeric_part = re.search(r'\d+', filtered_col).group()

                # Convert the numeric part to an integer
                numeric_value = int(numeric_part)

                # Create the new string with key and the numeric value
                new_string = f'{key}{numeric_value}'
                
                # rename the columns
                df = df.rename(columns={filtered_col:new_string})
                
                # append to keep the new name of the column
                col_name_list.append(new_string)
                
        elif len(filtered_cols) == 1: # only one column exists to be renamed
            
            # rename the existing columns
            df = df.rename(columns={filtered_cols[0]:key})
            col_name_list.append(key)
            
        else:
            
            print('there is no column name called ' + value + ' in the dataframe')
            
    # subset the input dataframe for given renamed columns
    if keep_renamed_only:
        df = df [col_name_list]
    
    # return
    return df
    


def Parameters (output_file,
                infile = None,
                soil_number = 12,
                soil_type = "usda",
                land_cover_number = 19,
                land_cover_type = "cec"):
    
    # 
    from .default_dict import General
    
    # write general
    for dictionary in General.values():
        write_dictionary(output_file, dictionary)
    
    # soil type
    if soil_type.lower() == 'usda':
        print('inside soil')
        from .default_dict import USDA
        print(USDA)
        # write general
        for dictionary in USDA.values():
            write_dictionary(output_file, dictionary, num = soil_number)
                
    # land cover
    if land_cover_type.lower() == 'cec':
        print('inside alnd cover')
        from .default_dict import CEC
        print(CEC)
        # write general
        for dictionary in CEC.values():
            write_dictionary(output_file, dictionary, num = land_cover_number)


def write_dictionary(output_file, dictionary, num = None):
    with open(output_file, 'a') as file:
        for key, value in dictionary.items():
            if key != 'section_head':
                if isinstance(value['value'], list):
                    if not num is None: # check the length
                        if num != len(value['value']):
                            sys.exit('the list is not with the same length as soil\
or land cover for key: '+ key)
                    values_line = '\t'.join(map(str, value['value']))
                    file.write(f"{key}\t{values_line}\t{value['comment']}\n")
                else:
                    file.write(f"{key}\t{value['value']}\t{value['comment']}\n")
            else:
                file.write(dictionary['section_head']+"\n")