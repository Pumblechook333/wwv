#%%
import grape as g
import os

#%%

def find_dirs_with_key_string(root_dir, key_string):
    """
    Finds directories containing a specified key string within a given root directory.

    Args:
        root_dir (str): The path to the directory to start searching from.
        key_string (str): The string to search for in directory names.

    Returns:
        list: A list of full paths to the directories that contain the key string.
    """
    matching_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if key_string in dirname:
                matching_dirs.append(os.path.join(dirpath, dirname))
    return matching_dirs

#%%

def order_list_by_keywords(dirs_list, keywords_list):
    """
    Orders a list of directories based on an equal-length list of keywords.

    Args:
        dirs_list (list): The list of directory strings to be sorted.
        keywords_list (list): The list of keywords that determines the final order.
                              Must be the same length as dirs_list.

    Returns:
        list: A new list with the directories sorted according to the keywords.
    """
    if len(dirs_list) != len(keywords_list):
        raise ValueError("The directories list and keywords list must be of equal length.")

    # Create a dictionary to map each keyword to its desired order
    order_map = {keyword: i for i, keyword in enumerate(keywords_list)}

    # Use the sorted() function with a custom key
    # The key function finds the keyword in the directory and returns its order value
    return sorted(dirs_list, key=lambda d: next((order_map[k] for k in keywords_list if k in d), len(keywords_list)))



#%%

NJ_data = 'DATA/bulk_beacon_data/NJ_data'
dirs_2022 = find_dirs_with_key_string(NJ_data, '2022')
print(dirs_2022)

#%%
keywords = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 
            'aug', 'sep', 'oct', 'nov', 'dec']
sorted_dirs_2022 = order_list_by_keywords(dirs_2022, keywords)
print(sorted_dirs_2022)

fname = 'antialias_2022'

#%%
ss = 60*10
nyquist = (1/ss)/2

grapes_2022 = g.GrapeHandler(sorted_dirs_2022, filt=True, comb=True, med=False, tShift=False, 
                             n=ss, filterorder=3, cutofffrequency=nyquist)
g.pickle_grape(grapes_2022, fname)

print('Done')

# %%

year = g.unpickle_grape(fname + '.pkl')
year.yearDopPlot(fname)

# %%
