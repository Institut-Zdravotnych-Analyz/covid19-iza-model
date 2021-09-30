import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import olefile


def save_df(df, name, path = None, replace = True, **kwargs):
	"""
	Saves pandas dataframe into .xls, .csv, .dta files
	@params:
		df   		- Required  : pandas dataframe object
		name    	- Required  : name with file extension (Str)
		path 	    - Optional  : path to save folder (Str)
		replace	    - Optional  : if true, will overwrite existing file with the same file name
		kwargs  	- Optional  : passed to individual save functions
	"""
	#path = os.path.join(project_dir, 'dirtyDB.xlsx')
	#writer = pd.ExcelWriter(path, engine = 'xlsxwriter')
	#df.to_excel(writer, sheet_name = 'dirtyDB')
	def find_free_f_name(name, path):
		f_path = os.path.join(path, name)
		if not os.path.exists(f_path):
			return f_path
		else:
			return find_free_f_name('.'.join(name.split('.')[:-1])+'1.'+name.split('.')[-1], path)
	
	if not path:
		path = os.path.dirname(os.path.abspath(__file__))
	f_path = os.path.join(path, name)
	
	if os.path.exists(f_path) and replace:
		os.remove(f_path)
	elif os.path.exists(f_path) and (not replace):
		f_path = find_free_f_name(name, path)

	if name.split('.')[-1][:3] == 'xls':
		df.to_excel(f_path, **kwargs)
	elif name.split('.')[-1][:3] == 'csv':
		df.to_csv(path_or_buf = f_path, **kwargs)
	elif name.split('.')[-1][:3] == 'dta':
		df.to_stata(f_path, **kwargs)
	else:
		print('unknown file extension:', name)

def load_df(name, path = None, **kwargs):
	"""
	Loads .xls, .csv, .dta files into pandas dataframe
	@params:
		name    	- Required  : name with file extension (Str)
		path 	    - Optional  : path to save folder (Str)
		kwargs  	- Optional  : passed to individual save functions
	"""
	if not path:
		if len(name.split('\\')) > 1:
			path = '\\'.join(name.split('\\')[:-1])
		else:
			path = os.path.dirname(os.path.abspath(__file__))	
	f_path = os.path.join(path, name)
	if name.split('.')[-1][:4] == 'xlsb':
		df = pd.read_excel(f_path, engine='pyxlsb',**kwargs)
	elif name.split('.')[-1][:3] == 'xls':
		try:
			df = pd.read_excel(f_path, **kwargs)
		except:
			ole = olefile.OleFileIO(f_path)
			if ole.exists('Workbook'):
				d = ole.openstream('Workbook')
				df = pd.read_excel(d, engine='xlrd')
	elif name.split('.')[-1][:3] == 'csv':
		df = pd.read_csv(f_path, **kwargs)
	elif name.split('.')[-1][:3] == 'dta':
		df = pd.read_stata(f_path, **kwargs)
	else:
		print('unknown file extension:', name)	
	return df


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()



def preset_pandas(max_rows=100, max_cols=200, disp_width=320):
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_cols)
    pd.set_option('display.width', disp_width)
    pd.options.mode.use_inf_as_na = True
    
def join_dfs(df, df_dict, on='', cols=[], add_on_prefix = True):
    df_dict = df_dict[cols]
    if add_on_prefix:
        cols = [str(on)+'_'+col for col in cols]
    if on in cols: df_dict.columns = cols
    else: df_dict.columns = [on] + cols[1:]
    
    df = df.join(df_dict.set_index(on), how='left', on=on)
    return df

def round_to_x_cifers_after_nonzero(df, col, n):
    df['num_text'] = df[col].astype(str).str.replace('.','')
    df['dot_pos'] = df[col].astype(str).str.split('.').str[0].str.len()
    df['lead'] = df['num_text'].str.split(r'1|2|3|4|5|6|7|8|9').str[0]
    df['strip'] = df['num_text'].str.replace(r'^0+','')
    df['num'] = df['strip'].str[:n].astype(int)
    df['oldnum'] = df['num'].values
    df.at[df['strip'].str[n].fillna('0').astype(int)>=5, 'num'] =df['num'] + 1
    df['dot_pos'] = df['dot_pos'] + df['num'].astype(str).str.len() - df['oldnum'].astype(str).str.len()
    df['trailing0s'] = [(x-n)*'0' for x in df['dot_pos']]
    df['numstr'] = df['lead'] + df['num'].astype(str) + df['trailing0s']
    df[col] = [x[1][:x[0]]+'.'+x[1][x[0]:] for x in df[['dot_pos','numstr']].values]
    df[col] = df[col].astype(float)
    df.drop(['num_text','dot_pos','lead','strip','oldnum','num','trailing0s','numstr'], axis = 1, inplace = True)
    return df




def plot_hists(df, cols, version='together', n_bins=10):
    if version == 'together':
        subplot_n = len(cols)
        fig, axs = plt.subplots(1, subplot_n, sharey=True, tight_layout=True,figsize=(16, 4))
        
        for i, column_name in enumerate(cols):
            axs[i].hist(df[column_name], bins=n_bins)
        plt.show()
        
    elif version == 'separate':
        for i,column_name in enumerate(cols):
            plt.hist(df[column_name], bins=n_bins, alpha=0.5, label=column_name)
        plt.legend(loc='best')
        plt.show()


def time_stuff():
    #leaps = [2008, 2012, 2016, 2020, 2024, 2028]
    years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    #days_in_months = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days_in_months_cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    #days_in_months_leap = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days_in_months_leap_cum = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
    hols_in_months_dict = {
        '2013': [[1,6],[],[29 ],[1 ],[1,8],[],[5],[29],[1,15],[],[1,17],[24,25,26]],
        '2014': [[1,6],[],[],[18,21],[1,8],[],[5],[29],[1,15],[],[1,17],[24,25,26]],
        '2015': [[1,6],[],[],[3 ,6 ],[1,8],[],[5],[29],[1,15],[],[1,17],[24,25,26]],
        '2016': [[1,6],[],[25,28],[],[1,8],[],[5],[29],[1,15],[],[1,17],[24,25,26]],
        '2017': [[1,6],[],[],[14,17],[1,8],[],[5],[29],[1,15],[],[1,17],[24,25,26]],
        '2018': [[1,6],[],[30 ],[2 ],[1,8],[],[5],[29],[1,15],[],[1,17],[24,25,26]],
        '2019': [[1,6],[],[],[19,22],[1,8],[],[5],[29],[1,15],[],[1,17],[24,25,26]],
        '2020': [[1,6],[],[],[10,13],[1,8],[],[5],[29],[1,15],[],[1,17],[24,25,26]],
        '2021': [[1,6],[],[],[2 ,5 ],[1,8],[],[5],[29],[1,15],[],[1,17],[24,25,26]],
        '2022': [[1,6],[],[],[15,18],[1,8],[],[5],[29],[1,15],[],[1,17],[24,25,26]],
        '2023': [[1,6],[],[],[7 ,10],[1,8],[],[5],[29],[1,15],[],[1,17],[24,25,26]]}
    first_weekend_dict = {
        '2013': [5,6],
        '2014': [4,5],
        '2015': [3,4],
        '2016': [2,3],
        '2017': [1,7],
        '2018': [6,7],
        '2019': [5,6],
        '2020': [4,5],
        '2021': [2,3],
        '2022': [1,2],
        '2023': [1,7]}
    return years, days_in_months_cum, days_in_months_leap_cum, hols_in_months_dict, first_weekend_dict

def work_days_in_year(year):
    _, days_in_months_cum, days_in_months_leap_cum, hols_in_months_dict, first_weekend_dict = time_stuff()
    if year%4 == 0:
        DinM = days_in_months_leap_cum
        days_tot = 366
    else:
        DinM = days_in_months_cum
        days_tot = 365
    df = pd.DataFrame(np.arange(1,days_tot+1))
    df.columns = ['days']
    df['work_day'] = 1
    df.at[(df['days']-first_weekend_dict[str(year)][0])%7 == 0, 'work_day'] = 0
    df.at[(df['days']-first_weekend_dict[str(year)][1])%7 == 0, 'work_day'] = 0
    holidays = [[idk+days for idk in hols_in_months_dict[str(year)][i]] for i,days in enumerate(DinM[:-1])]
    holidays = [item for sublist in holidays for item in sublist]
    for holiday in holidays:
        if (not (holiday - first_weekend_dict[str(year)][0])%7 == 0) and \
            (not (holiday - first_weekend_dict[str(year)][1])%7 == 0):
            df.at[df['days'] == holiday, 'work_day'] = 0
    print(f"Total days = {df['days'].max()} in {year}")
    print(f"Work days = {df['work_day'].sum()}, holidays + weekends = {df['days'].max()-df['work_day'].sum()}")
    return df
    

       
def date_to_days(df, date_col='DATUM', work_days=False, start=None):
    years, days_in_months_cum, days_in_months_leap_cum, hols_in_months_dict, first_weekend_dict = time_stuff()
    
    days_col = 'DAYS_'+date_col
    str_date_col = 'str'+date_col
    
    df[str_date_col] = df[date_col].astype(str)
    df[days_col] = df[str_date_col].str[4:6].astype(int)
    df[days_col] = df[days_col] - 1
    
    df['int_day_col'] = df[str_date_col].str[6:].astype(int)
    df['int_year_col'] = df[str_date_col].str[:4].astype(int)
    df_years = df['int_year_col'].unique()
    df_years = np.sort(df_years)
    
    for year in df_years:
        if not year in years: print('dataset has years date_to_days function is not ready for')
    if not start is None:
        if start < df_years[0]:
            years = years[years.index(start):years.index(df_years[-1])+1]
        else: print('date_to_days function is not ready to start later then the year in dataset, only sooner')
    else:
        years = years[years.index(df_years[0]):years.index(df_years[-1])+1]
        
    days_in_years = [0]
    for year in years:
        if year%4 == 0: days_in_years.append(366)
        else: days_in_years.append(365)
    days_in_years = [sum(days_in_years[:i+1]) for i in range(len(days_in_years))]
    
    for i, year in enumerate(years):
        if year in df_years:
            loca = df['int_year_col'] == int(year)
            print(year, i, loca.sum())
            if year%4 == 0:
                DinM = days_in_months_leap_cum
            else:
                DinM = days_in_months_cum
            df.at[loca, days_col] = df.loc[loca][days_col].apply(lambda x: DinM[x])
            df.at[loca, days_col] = df.loc[loca][days_col] + df.loc[loca]['int_day_col'] + days_in_years[i]
            if work_days:
                df['WORK_DAY'] = 1
                df.at[(loca) & ((df[days_col]-first_weekend_dict[str(year)][0])%7 == 0), 'WORK_DAY'] = 0
                df.at[(loca) & ((df[days_col]-first_weekend_dict[str(year)][1])%7 == 0), 'WORK_DAY'] = 0
                holidays = [[idk+days for idk in hols_in_months_dict[str(year)][i]] for i,days in enumerate(DinM[:-1])]
                holidays = [item for sublist in holidays for item in sublist]
                for holiday in holidays:
                    if (not (holiday - first_weekend_dict[str(year)][0])%7 == 0) and \
                        (not (holiday - first_weekend_dict[str(year)][1])%7 == 0):
                        df.at[(loca) & (df[days_col] == holiday), 'WORK_DAY'] = 0
            
    df = df.drop([str_date_col,'int_year_col','int_day_col'], axis=1)
    return df



def seven_davg(arr):
    dtype = arr.dtype
    ex_arr = np.zeros(arr.shape[0]+7)
    avg_arr = np.zeros(arr.shape[0])
    ex_arr[:3] = arr[0]
    ex_arr[-4:] = arr[-1]
    ex_arr[3:-4] = arr
    for i in range(7):
        avg_arr += ex_arr[i:-7+i]
    avg_arr /= 7
    avg_arr = avg_arr.astype(dtype)
    return avg_arr



def myplot(*args,
           ptype= '',
           x_lab = '',
           y_lab = '',
           leg = [],
           **kwargs):
    if ptype == 'plot':
        plt.plot(*args,*kwargs)
    if ptype == 'hist':
        plt.hist(*args,*kwargs)
    if ptype == 'scatter':
        plt.scatter(*args,*kwargs)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    plt.legend(leg, loc='best')
    plt.grid(True)
    plt.show()	


def timeit(t=None, s='', interval=[-1,-2]):
    """
    Keeps track of time and prints time intervals
    @params:
        t         	- Required  : if None return [time.time()], otherwise should be a list to which time.time() is appended
        s       	- Optional  : string to be printed
        interval   	- Optional  : printed s is followed by t[interval[0]] - t[interval[1]]
	Returns:
        t           - list of times
    """ 
    if t is None:
        return [time.time()]
    else:
        t.append(time.time())
        print(f'{s} {(t[interval[0]] - t[interval[1]]):.2f}s')
        return t



















