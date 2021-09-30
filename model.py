import os
import help_functions as fn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import cProfile, pstats, io
from scipy import interpolate
import numexpr as ne
import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 500  # Set Duration To 1000 ms == 1 second


def profile(fnc):
	"""A decorator that uses cProfile to profile a function"""
	def inner(*args, **kwargs):
		pr = cProfile.Profile()
		pr.enable()
		retval = fnc(*args, **kwargs)
		pr.disable()
		s = io.StringIO()
		sortby = 'cumulative'
		ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
		ps.print_stats()
		with open('profile.txt', 'w+') as f:
			f.write(s.getvalue())
		#print(s.getvalue())
		return retval
	return inner

def reset_imunity(df, i, **kwargs):
    loc = df['Ia'] > 0
    date = df['I_date'].values
    I_diff = ne.evaluate("(i-date)")
    df.at[((loc) & (I_diff > 180)) ,'Ia'] = 0
    return df, I_diff

def interpolation(x,y, x_new):
    interp = interpolate.interp1d(x, y, kind = "linear", bounds_error=False, fill_value="extrapolate")
    return interp(x_new)


def log(df, I_diff, i, inf_data, df_res=None, df_dis_times=None, **kwargs):
    i_states_cols = ['Id', 'Ih', 'Is', 'Ia']
    df_sums = pd.DataFrame({'day':i},index=[0])
    
    if not (df_dis_times['hospital_s'].fillna(-1).unique()==[-1,8]).all():
        print('your lazy log function is broken 1')
        exit()
    if not (df_dis_times['hospital_e'].fillna(-1).unique()==[-1,17]).all():
        print('your lazy log function is broken 2')
        exit()
    #a = df['I_date_diff'].values Implement
    #loc_s = ne.evaluate('a >= 8')
    loc_s = I_diff >= 8
    loc_e = I_diff <= 17
    loc_Id = df['Id'] == 1
    
    df_sums['dead'] = ((loc_Id) & (~loc_e)).sum()
    df_sums['hosp'] = (((df['Ih']==1) | (loc_Id)) & (loc_s) & (loc_e)).sum()
    df_sums['vaccinated'] = df['vac'].sum()
    eh = df[i_states_cols].values #slow bitch
    df_sums['were_infected'] = ne.evaluate("sum(eh)") #slow bitch
    df_sums['inf_resistance'] = df['R_inf'].sum()
    df_sums['infectious'] = inf_data[0]
    df_sums['new_infections'] = inf_data[1]
    df_sums['new_future_hosp'] = inf_data[4]
    df_res = df_res.append(df_sums, ignore_index=True)
    return df_res


def vaccinate(df, i, df_vak=None, **kwargs):
    vak_cols = ['pfizer-1', 'moderna-1', 'astra-1', 'sputnik-1', 'jnj']
    resistance_cols = ['Rvp', 'Rvm', 'Rva', 'Rvs', 'Rvj']
    n_vac_tod_arr = np.zeros(6, dtype=int)
    n_vac_tod_arr[1:] = df_vak.loc[df_vak.date_i==i][vak_cols].values[0]
    n_vac_today = n_vac_tod_arr.sum()
    
    a = df.vac.values
    b = df.index.values
    c = ne.evaluate("a!=0")
    d = np.ma.masked_array(b, mask=c)
    ind_dup = np.random.choice(d, size=int(n_vac_today))
    ind = np.unique(ind_dup)
    vac_arr = np.zeros((len(ind), 7), dtype=int)
    n_vac_tod_arr = n_vac_tod_arr.cumsum()
    for n in range(5):
        vac_arr[n_vac_tod_arr[n]:n_vac_tod_arr[n+1],n] = 1 #resistance_col 
    vac_arr[:,5] = 1 # 'vac' column
    vac_arr[:,6] = i # 'R_date' column
    df.at[ind, resistance_cols+['vac', 'R_date']] = vac_arr
    return df

def get_Rvac(df, i, max_inf_prots=None, max_hosp_prots=None,
             R_post_inf=None, post_inf_full_R=None, df_vac_times=None, **kwargs):
    resistance_cols = ['Rvp', 'Rvm', 'Rva', 'Rvs', 'Rvj']
    #res_inf_prot_cols = [col+'_inf_prot' for col in resistance_cols]
    i_states_cols = ['Id', 'Ih', 'Is', 'Ia']

    Rdate = df['R_date'].values
    Idate = df['I_date'].values
    b = df[resistance_cols].values
    inf = df[i_states_cols].values
    
    R_diff = ne.evaluate("(i-Rdate)")
    R_diff = ne.evaluate('where(R_diff < 0, 0, R_diff)')
    I_diff = ne.evaluate("(i-Idate)")
    r_diff1 = np.repeat(R_diff[:,np.newaxis], 5, axis=1)
    #k_hosp_prots = np.array([df_vac_times['k_hosp_prot']],dtype='float32')
    #k_inf_prots = np.array([df_vac_times['k_inf_prot']],dtype='float32')
    k_inf_prots = np.array([max_inf_prots[0]/df_vac_times['protection_delay']],dtype='float32')
    k_hosp_prots = np.array([max_hosp_prots[0]/df_vac_times['protection_delay']],dtype='float32')
    
    r_diff2 = ne.evaluate('r_diff1 * b')
    inf_prots = ne.evaluate('r_diff2 * k_inf_prots')
    #hosp_prots = ne.evaluate('r_diff2 * k_hosp_prots')
    vac_r = ne.evaluate('max(where(inf_prots < max_inf_prots, inf_prots, max_inf_prots), axis = 1)')
    was_inf = ne.evaluate('max(inf, axis=1)')
    inf_r1 = ne.evaluate('where(I_diff > post_inf_full_R, R_post_inf, 1)')
    inf_r = ne.evaluate('where(was_inf > 0, inf_r1, 0)')
    
    df['R_inf'] = ne.evaluate('1 - (1-vac_r)*(1-inf_r)')
    #df['R_inf'] = ne.evaluate('where(vac_r > inf_r, vac_r, inf_r)')
    return df, I_diff

def infect_new(df, i, df_okr=None, df_NAR=None, R0=None, auto_r0=None, df_dis_times=None, daily_imports=0, slabsi_automat=0, **kwargs):
    if not df_dis_times['infectious_s'].unique()==4:
        print('your lazy infect_new function is broken 1')
        exit()
    if not (df_dis_times['infectious_e'].unique()==[6,8]).all():
        print('your lazy infect_new function is broken 2')
        exit()
    auto_reg = df_NAR.iloc[-2][['1','2','3','4','5','6','7']]/ df_NAR.iloc[-2][['1','2','3','4','5','6','7']].sum()
    if slabsi_automat == 1:
        if R0>5:
            auto_r0 = np.array([0.456, 0.42 , 0.390, 0.360, 0.33 , 0.2835 , 0.25 ])
        if i>107:
            #auto_r0 = np.array([0.52, 0.455 , 0.407, 0.368, 0.33 , 0.2835 , 0.25 ])
            auto_r0 = np.array([0.540, 0.490 , 0.450, 0.420, 0.39 , 0.2835 , 0.25 ])
    '''
    if slabsi_automat > 0:
        if R0>5 and R0<=6:
            auto_r0 = np.array([0.506, 0.44 , 0.396, 0.363, 0.33 , 0.2835 , 0.25 ])
        elif R0>6:
            auto_r0 = np.array([0.552, 0.48 , 0.432, 0.396, 0.355 , 0.2835 , 0.25 ])
            if slabsi_automat == 1:
                auto_r0 = np.array([0.506, 0.44 , 0.396, 0.363, 0.33 , 0.2835 , 0.25 ])
    auto_r0 =     np.array([0.46,  0.40,  0.36,  0.33,  0.30,  0.27,    0.25])
    '''
    R0_ratio = (auto_reg*auto_r0).sum()
    #print(auto_r0)
    
    a = df['I_date'].values
    diff = ne.evaluate('i-a')
    if R0<5:
        loc_s = ne.evaluate('diff >= 4')
        loc_e1 = ne.evaluate('diff <= 6')
        loc_e2 = ne.evaluate('diff <= 8')
        infectious_l = [5, 5, 5, 3]
    elif R0>5 and R0<=6:
        loc_s = ne.evaluate('diff >= 3')
        loc_e1 = ne.evaluate('diff <= 6')
        loc_e2 = ne.evaluate('diff <= 8')
        infectious_l = [6, 6, 6, 4]
    elif R0>6:
        loc_s = ne.evaluate('diff >= 2')
        loc_e1 = ne.evaluate('diff <= 6')
        loc_e2 = ne.evaluate('diff <= 8')
        infectious_l = [7, 7, 7, 5]
    R0 = R0_ratio*R0
    i_states_cols = ['Id', 'Ih', 'Is', 'Ia']
    loc_e_cols = [loc_e2, loc_e2, loc_e2, loc_e1]
    n_to_infect = 0 + daily_imports
    n_infectious = 0
    for j, col in enumerate(i_states_cols):
        n = ((df[col] == 1) & (loc_s) & (loc_e_cols[j])).sum()
        n_infectious += n
        #inf = (np.random.geometric(1/(R0/df_dis_times.loc[col]['infectious_l']+1), n) - 1).sum()
        inf = (np.random.geometric(1/(R0/infectious_l[j]+1), n) - 1).sum()
        n_to_infect += inf
        #if n > 0:
        #    print(col, n, inf)
    
    new_future_hosp = 0
    if n_to_infect>0:
        #Check resistances
        inf_ind_dup = np.random.choice(df.loc[df['Id']==0].index, size=n_to_infect)
        inf_ind = np.unique(inf_ind_dup) # tu stracam par ludi ale kaslat lebo inak je to pomale
        inf_rng = np.random.uniform(size=len(inf_ind))
        #df.at[inf_ind[:3], 'R_inf'] = 1
        temp = df.loc[inf_ind]
        inf_ind = temp[temp['R_inf'] < inf_rng].index 
        new_infections_okres = temp.loc[inf_ind].groupby(['lokalita_kod']).size()
        df_okr = df_okr.append(new_infections_okres, ignore_index=True)
        new_infections = inf_ind.shape[0]
        
        if new_infections>0:
            #Check which disease they get
            #I_arr = df.loc[inf_ind][['vac', 'Ia_prob', 'Is_prob', 'Ih_prob', 'Id_prob']].values
            I_arr = df.loc[inf_ind][['vac', 'Id_prob', 'Ih_prob', 'Is_prob', 'Ia_prob']].values
            I_prob_arr = I_arr[:,1:]
            I_vac_arr = I_arr[:,0]
            inf_rng = np.random.uniform(size=inf_ind.shape[0])
            I_ind_arr = (I_prob_arr > inf_rng[:,np.newaxis]).argmax(axis=1)
            I_ind_cor = np.where((I_ind_arr<2) & (I_vac_arr>0), 2, I_ind_arr)
            temp = np.zeros((new_infections,6), dtype='float32')
            for j in range(4):
                temp[:,j] = (I_ind_cor == j).astype('float32')
            temp[:,-2] = i
            new_future_hosp = temp[:,0:2].sum()
            df.at[inf_ind, i_states_cols+['I_date','I_date_diff']] = temp
    else:
        new_infections=0
    return df, df_okr, [n_infectious, new_infections, R0_ratio, R0, new_future_hosp]

def automat(df, i, df_res=None, df_NAR=None, df_okr=None, df_dem=None, test_ratio=None,
            new_inf_delay=None, df_auto=None, **kwargs):
    crit_cols = ['NAR_hosp', 'NAR_7inc_PCRAG', 'NAR_PCR_TPR', 'NAR_Rt']
    min_crit_cols = [i+'_min' for i in crit_cols]
    #dummy incidencia
    inci = df_res['new_infections'].iloc[-new_inf_delay-7:-new_inf_delay].sum()/test_ratio/len(df)*100000
    okr_inci = (df_okr.fillna(0).iloc[-new_inf_delay-7:-new_inf_delay].sum()/test_ratio/df_dem['spolu'].values*100000).values
    hosp = df_res['hosp'].iloc[-7:].sum()/7
    #TPR = df_res['pos_tests'].iloc[-7:].sum()/df_res['tests'].iloc[-7:].sum()
    TPR = 0.
    Rt = 2.
    crit_val_arr = np.array([hosp, inci, TPR, Rt])
    #print(crit_val_arr)
    crit_min = df_auto[min_crit_cols].iloc[:2:-1].values
    crit_spln = np.zeros(7)
    temp = crit_min <= crit_val_arr[np.newaxis,:]
    crit_spln[:4] = (temp).sum(axis=1)
    crit_spln[6] = 3
    host_crit_level = np.zeros(7)
    host_crit_level[:4] = (temp)[:,0]
    host_crit_level[6] = 1
    df_crits = pd.DataFrame({'splnene': crit_spln, 'stupen': [7,6,5,4,3,2,1], 'hosp_crit':host_crit_level},
                            index=np.arange(0,7))
    #print(df_crits)
    lvl_hosp = df_crits.loc[df_crits['hosp_crit']==1]['stupen'].max()
    lvl_3of4 = df_crits.loc[df_crits['splnene']>=3]['stupen'].max()
    lvl_true_prev1 = df_NAR['stupen'].iloc[-1]
    lvl_true_prev2 = df_NAR['stupen'].iloc[-2]
    lvl_calc_prev1 = df_NAR['stupen_calc'].iloc[-1]
    lvl_calc_prev2 = df_NAR['stupen_calc'].iloc[-2]
    
    lvl_calc = max(lvl_hosp, lvl_3of4)
    if lvl_calc_prev2 < lvl_true_prev1: 
        if lvl_calc_prev1 <= lvl_calc_prev2 and lvl_true_prev1 == lvl_true_prev2:
            lvl_nar = lvl_true_prev1 - 1
        else:
            lvl_nar = lvl_true_prev1
    elif lvl_calc_prev1 > lvl_true_prev1:
        lvl_nar = lvl_calc_prev1
    else:
        lvl_nar = lvl_true_prev1
    if lvl_nar<4:
        lvl_nar = 1
        
    df_NAR = df_NAR.append(pd.DataFrame({'stupen':lvl_nar,'stupen_calc':lvl_calc,'i':i},index=[0]), ignore_index=True)
    
    # Okresne stupne
    inci_arr = np.ones((len(df_dem),7))
    inci_arr *= df_auto.REG_7inc_PCRAG_min
    okr_inci = (df_okr.fillna(0).iloc[-new_inf_delay-7:-new_inf_delay].sum()/test_ratio/df_dem['spolu'].values*100000).values
    okr_lvls = (inci_arr <= okr_inci[:,np.newaxis]).sum(axis=1)
    okr_lvls[np.where(okr_lvls<lvl_nar)] = lvl_nar
    for n in range(1,8):
        df_NAR.at[df_NAR.index[-1], str(n)] = (okr_lvls == n).sum()
        
    return df_NAR 

def novy_automat_init(df_okr=None, df_dem=None, new_inf_delay=None, test_ratio=None, **kwargs):
    
    okr_inci = (df_okr.fillna(0).iloc[-new_inf_delay-21:-new_inf_delay-7].sum()/test_ratio/df_dem['spolu'].values*100000)
    df_okr_inci = pd.DataFrame(okr_inci).transpose()
    okr_inci = (df_okr.fillna(0).iloc[-new_inf_delay-14:-new_inf_delay-7].sum()/test_ratio/df_dem['spolu'].values*100000)
    df_okr_inci = df_okr_inci.append(pd.DataFrame(okr_inci).transpose(), ignore_index=True)
    okr_inci = (df_okr.fillna(0).iloc[-new_inf_delay-7:-new_inf_delay].sum()/test_ratio/df_dem['spolu'].values*100000)
    df_okr_inci = df_okr_inci.append(pd.DataFrame(okr_inci).transpose(), ignore_index=True)
    
    df_true_lvls = pd.DataFrame(okr_inci).transpose()
    df_true_lvls.append(pd.DataFrame(okr_inci).transpose())
    df_true_lvls.append(pd.DataFrame(okr_inci).transpose())
    df_true_lvls[:] = 1
    return df_true_lvls, df_okr_inci

def novy_automat(i, df_NAR=None, df_okr=None, df_dem=None, test_ratio=None,
                new_inf_delay=None, df_auto_new=None, df_true_lvls=None, 
                df_okr_inci=None, zaock_stupne=None, **kwargs):
    #test_ratio = 2.8
    #new_inf_delay = 6
    okr_inci = (df_okr.fillna(0).iloc[-new_inf_delay-7:-new_inf_delay].sum()/test_ratio/df_dem['spolu'].values*100000)
    df_okr_inci = df_okr_inci.append(pd.DataFrame(okr_inci).transpose(), ignore_index=True)
    '''
    okr_inci = (df_okr[[1,2,3,4,5]].fillna(0).iloc[-new_inf_delay-7:-new_inf_delay].sum()/test_ratio/df_dem['spolu'].iloc[:5].values*100000)
    okr_inci[1] = okr_inci[1]*2
    okr_inci[5] = okr_inci[5]*3
    df_okr_inci = pd.DataFrame(okr_inci).transpose()
    df_okr_inci = df_okr_inci.append(pd.DataFrame(okr_inci).transpose()*1.15, ignore_index=True)
    df_okr_inci = df_okr_inci.append(pd.DataFrame(okr_inci).transpose()*1.3, ignore_index=True)
    df_okr_inci = df_okr_inci.append(pd.DataFrame(okr_inci).transpose()*1.3, ignore_index=True)
    df_okr_inci = df_okr_inci.append(pd.DataFrame(okr_inci).transpose()*1.2, ignore_index=True)
    df_okr_inci = df_okr_inci.append(pd.DataFrame(okr_inci).transpose()*1.5, ignore_index=True)
    df_okr_inci = df_okr_inci.append(pd.DataFrame(okr_inci).transpose()*0.9, ignore_index=True)
    df_okr_inci = df_okr_inci.append(pd.DataFrame(okr_inci).transpose()*0.2, ignore_index=True)
    df_okr_inci = df_okr_inci.append(pd.DataFrame(okr_inci).transpose()*0., ignore_index=True)
    df_okr_inci.columns.name=None'''
    rast = df_okr_inci.iloc[1:].values >= df_okr_inci.iloc[:-1].values
    # Find OG levels
    okr_lvls_arr = df_okr_inci[1:].copy().values
    okr_lvls_arr[:] = 1
    for n in range(4,0,-1):
        min_rast = df_auto_new['rast_7inc_min'].iloc[n]
        max_rast = df_auto_new['rast_7inc_max'].iloc[n]
        min_kles = df_auto_new['kles_7inc_min'].iloc[n]
        max_kles = df_auto_new['kles_7inc_max'].iloc[n]
        okr_lvls_arr = np.where((rast==True) & (df_okr_inci.iloc[1:]>=min_rast) & (df_okr_inci.iloc[1:]<max_rast), n+1, okr_lvls_arr)
        okr_lvls_arr = np.where((rast==False) & (df_okr_inci.iloc[1:]>=min_kles) & (df_okr_inci.iloc[1:]<max_kles), n+1, okr_lvls_arr)
    
    # Define true levels based on switching mechanisms
    #df_true_lvls = pd.DataFrame(okr_lvls_arr[-1].reshape([1,5]), columns=okr_inci.index)
    lvl_rast = okr_lvls_arr[-1] >= df_true_lvls.values[-1]
    true_lvls = np.zeros(len(okr_inci))
    true_lvls = np.where(lvl_rast, okr_lvls_arr[-1], true_lvls)
    true_lvls = np.where((~lvl_rast), df_true_lvls.values[-1], true_lvls)
    true_lvls = np.where((~lvl_rast) & (okr_lvls_arr[-1]==okr_lvls_arr[-2]), okr_lvls_arr[-1], true_lvls)
    true_lvls = np.where((~lvl_rast) & (~rast[-2]) & (~rast[-3]), okr_lvls_arr[-1], true_lvls)
    true_lvls = true_lvls - zaock_stupne.loc[i].values
    #true_lvls = true_lvls - zaock_stupne.loc[i].values[:5]
    true_lvls = np.where(true_lvls<1, 1, true_lvls)
    df_true_lvls = df_true_lvls.append(pd.DataFrame(true_lvls, index=okr_inci.index).transpose(), ignore_index=True)
    #print(df_okr_inci)
    #print(rast)
    #print(okr_lvls_arr)
    #print(true_lvls)
    df_NAR = df_NAR.append(pd.DataFrame({'stupen':0,'stupen_calc':0,'i':i},index=[0]), ignore_index=True)
    for n in range(1,8):
        df_NAR.at[df_NAR.index[-1], str(n)] = (true_lvls == n).sum()    
    
    
    #84    84 2021-08-09 novy automat schvaleny
    #91    91 2021-08-16 novy automat platny
    return df_NAR, df_true_lvls, df_okr_inci

def ind_var(var1, var2, days, 
            i_start=110,
            t_transition=40):
    arr = np.ones(days)
    arr *= var1
    arr[int(i_start+t_transition):] = var2
    arr[int(i_start):int(i_start+t_transition)] += np.linspace(0,1,t_transition)*(var2-var1)
    return arr



#@profile
def sim(days=None,
        df=None,
        R0s=None,
        max_inf_prots_in_time=None,
        max_hosp_prots_in_time=None,
        df_okr=None,
        df_NAR=None,
        df_res=None,
        max_inf_prots=None,
        max_hosp_prots=None,
        df_inf_ind_probs=None,
        **kwargs):


    fn.printProgressBar(0, days, prefix = 'Sim', suffix = f'{0}/{days} days', length = 25)
    first=True
    for i in range(days):
        R0 = R0s[i]
        max_inf_prots[:,:] = max_inf_prots_in_time[i,:].reshape((5))
        max_hosp_prots[:,:] = max_hosp_prots_in_time[i,:].reshape((5))

        if R0 > 5.5 and first:
            df.drop(['Id_prob', 'Ih_prob', 'Is_prob', 'Ia_prob'], axis = 1, inplace = True)
            df = fn.join_dfs(df, df_inf_ind_probs, on='vek', cols=['vek', 'Id_prob', 'Ih_prob', 'Is_prob', 'Ia_prob'], add_on_prefix=False)
            df[['Id_prob', 'Ih_prob', 'Is_prob', 'Ia_prob']] = df[['Id_prob', 'Ih_prob', 'Is_prob', 'Ia_prob']].cumsum(axis=1)
            first=False
        #df['Rvp_inf_prot_max'] = df_R.iloc[i]['RvpESC']
        #df['Rvm_inf_prot_max'] = df_R.iloc[i]['RvmESC']
        #df['Rva_inf_prot_max'] = df_R.iloc[i]['RvaESC']
        #df['Rvs_inf_prot_max'] = df_R.iloc[i]['RvsESC']
        #df['Rvj_inf_prot_max'] = df_R.iloc[i]['RvjESC']
        #df, I_diff = reset_imunity(df, i, **kwargs)
        #df = vaccinate(df, i, **kwargs)
        df, I_diff = get_Rvac(df, i, max_inf_prots=max_inf_prots, max_hosp_prots=max_hosp_prots, **kwargs) #max_inf_prots, max_hosp_prots, R_post_inf, post_inf_full_R
        df, df_okr, inf_data = infect_new(df, i, df_okr=df_okr, df_NAR=df_NAR, R0=R0, **kwargs)
        df_res = log(df, I_diff, i, inf_data, df_res=df_res, **kwargs)
        if i==84:
            df_true_lvls, df_okr_inci = novy_automat_init(df_okr=df_okr, **kwargs)
        if i%7==0:
            #print(i)
            if i < 91:
                df_NAR = automat(df, i, df_res=df_res, df_NAR=df_NAR, df_okr=df_okr, **kwargs)
            else:
                df_NAR, df_true_lvls, df_okr_inci = novy_automat(i, df_NAR=df_NAR, df_okr=df_okr, 
                                                                 df_true_lvls=df_true_lvls, 
                                                                 df_okr_inci=df_okr_inci, **kwargs)

        fn.printProgressBar(i+1, days, prefix = 'Sim', suffix = f'{i+1}/{days} days', length = 25)
        #if i >20: 		
        #    break
    return df, df_res, df_NAR, df_okr, df_true_lvls, df_okr_inci


if __name__ == '__main__':
    times = fn.timeit(t=None, s='', interval=[-1,-2])
    # Check to set up directory paths and pandas presets in Andrej's notebook
    if os.path.realpath(__file__).split('\\')[2]=='klukaa':
        fn.preset_pandas(max_rows=250, max_cols=50, disp_width=320)
        path = 'C:\\Users\\klukaa\\Desktop\\sim\\'
    demografia_dir = path + 'demografia'
    vakciny_dir = path + 'odhad-vakciny\\data'
    params_dir = path + 'params_data'
    testovanie_dir = path + 'testing_data'
    hosp_dir = path + 'hospitalizacie_umrtia'
    times = fn.timeit(t=None, s='', interval=[-1,-2])
    np.random.seed(1)

    '''
    znizit efektivitu ostatnych vakcin
    
    
    peak probability na serial interval 4 dni R0 5-8
    https://www.medrxiv.org/content/10.1101/2021.06.04.21258205v1.full-text
    23.7 % to 25.5% 20-40 hosp chance increase
    https://en.wikipedia.org/wiki/SARS-CoV-2_Delta_variant
    6% from 4% '''    
    
    def full_sim(times,
                 vak_choice=None,
                 ind_start_date=None,
                 slabsi_automat=0,
                 to_plot=True
                 ):
        
        ind_start_dates = {
            'July':45,
            'mid-July':55,
            #'mid-July':60,
            'August':76,
            'September':107
            }
        R0 = 4.175
        test_ratio = 2.8
        start_date = pd.to_datetime("2021-5-17")   
        new_inf_delay = 6 #incubation period + reporting delay
        days = 229
        daily_imports = 20
        ind_start = ind_start_dates[ind_start_date] # Kedy zacne community spread indickeho variantu
        t_transition = 40 #days
        post_inf_full_R = 60 #days
        auto_r0 = np.array([0.46, 0.40, 0.36, 0.33, 0.30, 0.27, 0.25])
        title = f'Vacc. max.: {vak_choice}%, delta starts with {ind_start_date} + slightly weaker automat'
        title = f'Vacc. max.: {vak_choice}%, delta starts with {ind_start_date} + weaker automat'
        R0s = ind_var(R0, 6.7, days, i_start=ind_start, t_transition=t_transition)    
        ind_var_hosp_rate = 1.08
        ind_var_hosp_rate = 1.13


        # Load data
        df_dem = fn.load_df('vekove-skupiny-okresy_cut.xlsx', path=demografia_dir)
        #['lokalita_kod', 'lokalita', 'spolu', '0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100-104', '105-109', '110-115']
        #df_dem_vak = fn.load_df('vekove-skupiny-vakciny.xlsx', path=demografia_dir)
        #['Okres', 'Očkovanosť (1. dávka)', '0-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+', 'Spolu', 'Počet obyvateľov']
        df_vek = fn.load_df('vekove-skupiny.xlsx', path=demografia_dir)
        #['vek', 'vek_skup_5r', 'vek_skup_vak']
        '''df_vak_n = fn.load_df('doses_predicted.csv', path=vakciny_dir)
        df_vak_5 = fn.load_df('doses_predicted_50p.csv', path=vakciny_dir)
        df_vak_6 = fn.load_df('doses_predicted_60p.csv', path=vakciny_dir)
        df_vak_7 = fn.load_df('doses_predicted_70p.csv', path=vakciny_dir)
        df_vak_8 = fn.load_df('doses_predicted_80p.csv', path=vakciny_dir)
        df_vak_dic = {
            'norm':df_vak_n,
            '50':df_vak_5,
            '60':df_vak_6,
            '70':df_vak_7,
            '80':df_vak_8}
        df_vak = df_vak_dic[vak_choice]'''
        #['date', 'moderna-1', 'pfizer-1', 'astra-1', 'moderna-2', 'pfizer-2', 'astra-2', 'jnj', 'sputnik-1', 'sputnik-2', 'moderna-capacity', 'pfizer-capacity', 'astra-capacity', 'jnj-capacity', 'sputnik-capacity', 'protected', 'protected_cumulative']
        #df_idk = fn.load_df('clinical_susceptibility_params.xlsx', path=params_dir)
        #['vek_skup_params1', 'susceptibility', 'symptomatic']
        df_R = fn.load_df('R0.xlsx', path=params_dir)
        #['DATUM', 'R0v1', 'R0v2']
        df_auto = fn.load_df('automat.xlsx', path=params_dir)
        #['farba', 'kod_farby', 'popis', 'stupen', 'NAR_hosp_min', 'NAR_hosp_max', 'NAR_7inc_PCRAG_min', 'NAR_7inc_PCRAG_max', 'NAR_PCR_TPR_min', 'NAR_PCR_TPR_max', 'NAR_Rt_min', 'NAR_Rt_max', 'REG_7inc_PCRAG_min', 'REG_7inc_PCRAG_max']
        df_auto_new = fn.load_df('novy_automat.xlsx', path=params_dir)
        #['farba', 'kod_farby', 'popis', 'stupen', 'rast_7inc_min', 'rast_7inc_max', 'kles_7inc_min', 'kles_7inc_max']
        df_test = fn.load_df('df_testing.xlsx', path=testovanie_dir)
        #['DATUM', 'week', 'PCR_N', 'PCR_pos', 'PCR_neg', 'PCR_TPR', 'AG_N', 'AG_pos', 'AG_neg', 'AG_TPR', 'AG_N_pac', 'AG_PAC_pos', 'AG_PAC_neg']
        df_hosp = fn.load_df('hospitalizacie.csv', path=hosp_dir)
        #['Datum', '0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-100', '100-115']
        df_umrt = fn.load_df('umrtia.csv', path=hosp_dir)
        #['DATUM', 'week', 'PCR_N', 'PCR_pos', 'PCR_neg', 'PCR_TPR', 'AG_N', 'AG_pos', 'AG_neg', 'AG_TPR', 'AG_N_pac', 'AG_PAC_pos', 'AG_PAC_neg']
        df_age_params = fn.load_df('age_specific parameters.xlsx', path=params_dir)
        #['vek_skup_params', 'zdroj', 'avg_vek', 'hosp/sympt', 'hosp/infected', 'crit/hosp', 'death/crit', 'death/non-crit', 'susceptibility', 'symptomatic', 'Ia', 'Is', 'Ih', 'Id']
        df_dis_times = fn.load_df('disease_timelines.xlsx', path=params_dir)
        df_dis_times = df_dis_times.set_index('type')
        #['type', 'type_pop', 'symptoms_s', 'symptoms_e', 'infectious_s', 'infectious_e', 'hospital_s', 'hospital_e', 'AG_pos_s', 'AG_pos_e', 'PCR_pos_s', 'PCR_pos_e', 'immunity_s', 'immunity_e']
        df_vac_times = fn.load_df('vaccination_timelines.xlsx', path=params_dir)
        df_vac_times = df_vac_times.set_index('vac_type')
        #['vac_type_pop', 'doses', 'doses_delay', 'protection_delay', 'inf_protection', 'hosp_protection', 'k_inf_prot', 'k_hosp_prot']
        times = fn.timeit(t=times, s='Data loaded', interval=[-1,-2])
        #['pocet', 'vak', 'vek', 'lokalita', 'lokalita_kod', 'date']
        df_vac = fn.load_df('vak_df.csv', path=vakciny_dir)
        df_vac['date'] = pd.to_datetime(df_vac['date'], errors = 'coerce')
        
        #['date', 'astra-1', 'jnj', 'moderna-1', 'pfizer-1', 'sputnik-1']
        df_vac_agg = fn.load_df('vak_df_agg.csv', path=vakciny_dir)
        df_vac_agg['date'] = pd.to_datetime(df_vac_agg['date'], errors = 'coerce')
        times = fn.timeit(t=times, s='Data loaded', interval=[-1,-2])
        
        #'''
        #Vytvor obyvatelov s vekmi a vekovymi skupinami ['vek_skup_5r', 'vek_skup_vak', 'vek']
        veky = np.random.choice(df_vek.vek_skup_5r.unique().tolist(), size=int(df_dem['spolu'].sum()), \
                                p=(df_dem[df_vek.vek_skup_5r.unique().tolist()].sum()/df_dem[df_vek.vek_skup_5r.unique().tolist()].sum().sum()).values.tolist())
        df = pd.DataFrame({'vek_skup_5r':veky}, \
                           index = np.arange(0,int(df_dem['spolu'].sum())))
        vek_skup_vak_gr = df_vek.groupby(['vek_skup_5r'])['vek_skup_vak'].unique().reset_index()
        
        df = fn.join_dfs(df, vek_skup_vak_gr, on='vek_skup_5r', cols=['vek_skup_5r','vek_skup_vak'], add_on_prefix=False)
        df['vek_skup_vak'] = df['vek_skup_vak'].apply(lambda x: x[0])
        vek_skup_gr = df_vek.groupby(['vek_skup_5r'])['vek'].unique().reset_index()
        df = fn.join_dfs(df, vek_skup_gr, on='vek_skup_5r', cols=['vek_skup_5r','vek'], add_on_prefix=False)
        df['vek'] = df['vek'].apply(lambda x: np.random.choice(x))
        times = fn.timeit(t=times, s='Veky ready', interval=[-1,-2])
        
        
        #Pridaj okresy obyvatelov ['lokalita_kod', 'lokalita']]
        okresy = df_dem['lokalita_kod'].unique().tolist()
        df['lokalita_kod'] = np.nan
        for skup in df_vek.vek_skup_5r.unique().tolist():
            loc = df['vek_skup_5r'] == skup
            okresy_pre_skup = np.random.choice(okresy, size=loc.sum(), p=(df_dem[skup]/df_dem[skup].sum()).values.tolist())
            df.at[loc, 'lokalita_kod'] = okresy_pre_skup
        df = fn.join_dfs(df, df_dem, on='lokalita_kod', cols=['lokalita_kod','lokalita'], add_on_prefix=False)
        times = fn.timeit(t=times, s='Okresy ready', interval=[-1,-2])
        
        #fn.save_df(df, 'prep_df.csv', path=params_dir, index=False, replace=True) 
        #exit()
        #'''
        #df = fn.load_df('prep_df.csv', path=params_dir)
        times = fn.timeit(t=times, s='df prepped', interval=[-1,-2])
    
        # blank init
        df['I_date'] = np.nan
        df['I_date_diff'] = 0
        i_states_cols = ['Id', 'Ih', 'Is', 'Ia']
        for col in i_states_cols:
            df[col] = 0
            #df.at[np.random.choice(df.index, size=500, replace=False), col] = 1
        df['I_date'] = 0
        df['R_date'] = np.nan
        df['R_inf'] = 0
        df['R_post_inf'] = 0.8
        df.at[df['vek']>65, 'R_post_inf'] = 0.5
        R_post_inf = df['R_post_inf'].values.astype('float32')
        #                 Ra, Rs, Rvp, Rvm, Rva, Rvs, Rvj
        #df['R_source'] = [0,  0,  0,   0,   0,   0,   0]
        df['vac'] = False
        resistance_cols = ['Rvp', 'Rvm', 'Rva', 'Rvs', 'Rvj']
        for col in resistance_cols:
            df[col] = 0
            df[col+'_inf_prot_max'] = df_vac_times.loc[col]['inf_protection']
            df[col+'_hosp_prot_max'] = df_vac_times.loc[col]['hosp_protection']
        max_inf_prots_in_time = np.zeros((days, 5))
        max_hosp_prots_in_time = np.zeros((days, 5))
        for i, col in enumerate(resistance_cols):        
            max_inf_prots_in_time[:,i] = ind_var(df_vac_times.loc[col]['inf_protection'], 
                                                 df_vac_times.loc[col]['ind_inf_protection'], 
                                                 days, i_start=ind_start, t_transition=t_transition)    
            max_hosp_prots_in_time[:,i] = ind_var(df_vac_times.loc[col]['hosp_protection'], 
                                                 df_vac_times.loc[col]['ind_hosp_protection'], 
                                                 days, i_start=ind_start, t_transition=t_transition)    
    
        # jebkanie sa s infection probabilities
        ages = np.arange(0,116)
        inf_prob_cols = ['Id_prob', 'Ih_prob', 'Is_prob', 'Ia_prob']
        df_inf_probs = pd.DataFrame({'vek':ages}, index = ages) 
        for col in inf_prob_cols:
            df_inf_probs[col] = 0
        loc = df_age_params['zdroj'] == 1
        df_inf_probs['Id_prob'] = interpolation(df_age_params.loc[loc]['avg_vek'], df_age_params.loc[loc]['Id'], ages)
        df_inf_probs['Ih_prob'] = interpolation(df_age_params.loc[loc]['avg_vek'], df_age_params.loc[loc]['Ih'], ages)
        loc = df_age_params['zdroj'] == 2
        df_inf_probs['Ia_prob'] = interpolation(df_age_params.loc[loc]['avg_vek'], df_age_params.loc[loc]['Ia'], ages)
        df_inf_probs['Is_prob'] = 1 - df_inf_probs['Id_prob'] - df_inf_probs['Ih_prob'] - df_inf_probs['Ia_prob']
        df = fn.join_dfs(df, df_inf_probs, on='vek', cols=['vek', 'Id_prob', 'Ih_prob', 'Is_prob', 'Ia_prob'], add_on_prefix=False)
        df[['Id_prob', 'Ih_prob', 'Is_prob', 'Ia_prob']] = df[['Id_prob', 'Ih_prob', 'Is_prob', 'Ia_prob']].cumsum(axis=1)
        # infekcia f of (kontakt, infection chance(vector type, resistance type, resistance date))
        times = fn.timeit(t=times, s='DF created', interval=[-1,-2])
        
        # jebkanie sa s infection probabilities pre indicky variant
        ages = np.arange(0,116)
        inf_prob_cols = ['Id_prob', 'Ih_prob', 'Is_prob', 'Ia_prob']
        df_inf_ind_probs = pd.DataFrame({'vek':ages}, index = ages)
        for col in inf_prob_cols:
            df_inf_ind_probs[col] = 0
        loc = df_age_params['zdroj'] == 1
        df_inf_ind_probs['Id_prob'] = interpolation(df_age_params.loc[loc]['avg_vek'], df_age_params.loc[loc]['Id'], ages)*ind_var_hosp_rate
        df_inf_ind_probs['Ih_prob'] = interpolation(df_age_params.loc[loc]['avg_vek'], df_age_params.loc[loc]['Ih'], ages)*ind_var_hosp_rate
        loc = df_age_params['zdroj'] == 2
        df_inf_ind_probs['Ia_prob'] = interpolation(df_age_params.loc[loc]['avg_vek'], df_age_params.loc[loc]['Ia'], ages)
        df_inf_ind_probs['Is_prob'] = 1 - df_inf_ind_probs['Id_prob'] - df_inf_ind_probs['Ih_prob'] - df_inf_ind_probs['Ia_prob']
        df_inf_ind_probs = df_inf_ind_probs.astype('float32')
        '''
        df = fn.join_dfs(df, df_inf_ind_probs, on='vek', cols=['vek', 'Id_prob', 'Ih_prob', 'Is_prob', 'Ia_prob'], add_on_prefix=False)
        df[['Id_prob', 'Ih_prob', 'Is_prob', 'Ia_prob']] = df[['Id_prob', 'Ih_prob', 'Is_prob', 'Ia_prob']].cumsum(axis=1)
        # infekcia f of (kontakt, infection chance(vector type, resistance type, resistance date))
        times = fn.timeit(t=times, s='DF created', interval=[-1,-2])
        '''
        
        #Old Vaccination initialization
        '''
        # Initialization states for vaccination
        vak_cols = ['pfizer-1', 'moderna-1', 'astra-1', 'sputnik-1', 'jnj']
        df_vak[vak_cols] = df_vak[vak_cols].astype(int)
        #df_vak = df_vak[['date', 'moderna-1', 'pfizer-1', 'astra-1', 'jnj', 'sputnik-1']]
        df_vak['date'] = pd.to_datetime(df_vak['date'], errors = 'coerce')
        df_vak['past'] = df_vak['date'] < start_date
        df_vak['date_i'] = np.arange(0,len(df_vak)) - df_vak['past'].sum()
        #df['pdDatumVykonuDim'].dt.day_name().str[:3]
    
        df[resistance_cols+['vac', 'R_date']] = df[resistance_cols+['vac', 'R_date']].fillna(0).astype(int)
        n_vac_tod_arr = np.zeros(6, dtype=int)
        n_vac_tod_arr[1:] = df_vak.loc[df_vak.past==True][vak_cols].sum().values
        n_vac_today = n_vac_tod_arr.sum()
        ind = np.random.choice(df.loc[df.vac==0].index, size=int(n_vac_today), replace=False)
        vac_arr = np.zeros((n_vac_today, 7), dtype=int)
        n_vac_tod_arr = n_vac_tod_arr.cumsum()
        for n in range(5):
            vac_arr[n_vac_tod_arr[n]:n_vac_tod_arr[n+1],n] = 1    
            temp = df_vak.loc[df_vak.past==True][['date_i', vak_cols[n]]]
            r_dates = np.repeat(temp.date_i, temp[vak_cols[n]]).reset_index(drop=True).values
            np.random.shuffle(r_dates)
            vac_arr[n_vac_tod_arr[n]:n_vac_tod_arr[n+1],6] = r_dates
        vac_arr[:,5] = 1 # 'vac' column
        df.at[ind, resistance_cols+['vac', 'R_date']] = vac_arr
        print(df[resistance_cols].value_counts())
        print(df['vac'].sum())
        times = fn.timeit(t=times, s='Vaccination initialized', interval=[-1,-2])
        #'''
        
        '''
        # Initialize states for umrtia and hospitalizations UNFINISHED
        temp = df_umrt['Row Labels'].str.split('/').str
        df_umrt['DATUM'] = temp[2] +'-'+ temp[0] +'-'+ temp[1]
        df_umrt['DATUM'] = pd.to_datetime(df_umrt['DATUM'], errors = 'coerce')
        df_umrt['spolu'] = df_umrt[df_umrt.columns[1:-1]].sum(axis=1)
        
        temp = df_hosp['Datum'].str.split('/').str
        df_hosp['DATUM'] = temp[2] +'-'+ temp[0] +'-'+ temp[1]
        df_hosp['DATUM'] = pd.to_datetime(df_hosp['DATUM'], errors = 'coerce')
        df_hosp['spolu'] = df_hosp[df_hosp.columns[1:-1]].sum(axis=1)  
        hosp_l = df_dis_times.loc['Ih']['hospital_e']-df_dis_times.loc['Ih']['hospital_s']
        df_hosp['spolu_increments'] = df_hosp['spolu'] 
        df_hosp['spolu_increments'].iloc[1:] = df_hosp['spolu'].iloc[1:].values - df_hosp['spolu'].iloc[:-1].values
        df_hosp['spolu_increments'].iloc[hosp_l:] = df_hosp['spolu'].iloc[1:].values - df_hosp['spolu'].iloc[:-1].values
        df_hosp['spolu_increments'] = (df_hosp['spolu_increments']/(hosp_l)).round().astype('float32')
        '''
        
        # Initialize infections
        df_test = df_test.loc[(df_test['DATUM']>=pd.to_datetime("2020-11-01")) & (df_test['DATUM']<=start_date)].reset_index(drop=True)
        df_test['pos'] = (test_ratio*(df_test['PCR_pos'] + df_test['AG_pos'])).round()
        df_test['pos'] = fn.seven_davg(df_test['pos'].values)
        df_test['pos'] = df_test['pos'].round()
        df_test['date_i'] = -np.arange(0, len(df_test))[::-1]
        old_inf = int(df_test['pos'].sum())
        inf_ind = np.random.choice(df.loc[df.vac==0].index, size=old_inf, replace=False)
        temp = df.loc[inf_ind]
        #Check which disease they get
        I_arr = df.loc[inf_ind][['Id_prob', 'Ih_prob', 'Is_prob', 'Ia_prob']].values
        inf_rng = np.random.uniform(size=inf_ind.shape[0])
        I_ind_arr = (I_arr > inf_rng[:,np.newaxis]).argmax(axis=1)
        temp = np.zeros((old_inf,6), dtype='float32')
        for j in range(4):
            temp[:,j] = (I_ind_arr == j).astype('float32')
        tempi = df_test[['date_i', 'pos']]
        r_dates = np.repeat(tempi.date_i, tempi['pos']).reset_index(drop=True).values
        np.random.shuffle(r_dates)
        temp[:,-2] = r_dates
        df.at[inf_ind, i_states_cols+['I_date','I_date_diff']] = temp
        times = fn.timeit(t=times, s='Infections initialized', interval=[-1,-2])
    
        df['was_inf'] = df[['Id', 'Ih', 'Is']].max(axis=1)
        df_temp = df.loc[df['was_inf']==0].reset_index().groupby(['lokalita_kod','vek']).index.unique().reset_index()
        df_temp['indexes'] = df_temp['index']
        df_temp = df_temp.drop(['index'], axis=1)
        df_temp[['lokalita_kod','vek']] = df_temp[['lokalita_kod','vek']].astype(int)
        df_temp['u_index'] = df_temp['lokalita_kod']*1000 + df_temp['vek']
        df_temp2 = df_vac.groupby(['lokalita_kod','vek'])['pocet'].sum().reset_index()
        df_temp2['u_index'] = (df_temp2['lokalita_kod']*1000 + df_temp2['vek']).astype('int32')
        df_temp = fn.join_dfs(df_temp, df_temp2, on='u_index', cols=['u_index', 'pocet'], add_on_prefix=False)
        df_temp['pocet'] = df_temp['pocet'].fillna(0).astype(int)
        df_temp['max_pop'] = df_temp['indexes'].str.len()
        df_temp['rozdiel'] = df_temp['max_pop'] - df_temp['pocet']
        N_cant_vax = df_temp.loc[df_temp['rozdiel']<0]['rozdiel'].sum()
        print('I cant vaccinate this many people perfectly: ', N_cant_vax)
        df_temp.at[df_temp['pocet']>df_temp['max_pop'], 'pocet'] = df_temp['max_pop']
        df_temp['chosen'] = np.nan
    
        def idk_group(df):
            return np.random.choice(df['indexes'], df['pocet'], replace=False)
        df_temp['chosen'] = df_temp.apply(idk_group, axis=1)
        
        df_vac_exp = df_vac.loc[df_vac.index.repeat(df_vac['pocet'])].reset_index(drop=True)
        df_vac_dates = df_vac_exp.groupby(['lokalita_kod','vek'])[['date','vak']].agg(list).reset_index()
        df_vac_dates['N_dates'] = df_vac_dates['date'].str.len()
        df_vac_dates['u_index'] = (df_vac_dates['lokalita_kod']*1000 + df_vac_dates['vek']).astype('int32')
    
        df_temp = fn.join_dfs(df_temp, df_vac_dates, on='u_index', cols=['u_index', 'date', 'vak'], add_on_prefix=False)
        
        # Get rid of dates and vaccine names of those who could not have been vaccinated
        df_temp['date'] = df_temp['date'].apply(lambda d: d if isinstance(d, list) else [])
        df_temp['vak'] = df_temp['vak'].apply(lambda d: d if isinstance(d, list) else [])
        def idk_group(df):
            d={}
            d['date'] = df['date'][:df['pocet']]
            d['vak'] = df['vak'][:df['pocet']]
            return pd.Series(d)
        df_temp[['date','vak']] = df_temp.apply(idk_group, axis=1)
        chosen = [item for sublist in df_temp.chosen.values.tolist() for item in sublist]
        date = [item for sublist in df_temp.date.values.tolist() for item in sublist]
        vak = [item for sublist in df_temp.vak.values.tolist() for item in sublist]
        lokalita_kod = df_temp.loc[df_temp.index.repeat(df_temp['pocet'])]['lokalita_kod'].reset_index(drop=True).values.tolist()
        vek = df_temp.loc[df_temp.index.repeat(df_temp['pocet'])]['vek'].reset_index(drop=True).values.tolist()
        print(len(chosen), len(date), len(vak), len(lokalita_kod), len(vek))
        
        df_idk = pd.DataFrame({'chosen':chosen, 'date':date, 'vak':vak, 'lokalita_kod':lokalita_kod, 'vek':vek})
        df_idk['R_date'] = (df_idk['date'] - df_idk['date'].min()).dt.days
        df_idk['R_date'] -= (start_date - df_idk['date'].min()).days
        df_idk['vac'] = 1
        resistance_cols = ['Rvp', 'Rvm', 'Rva', 'Rvs', 'Rvj']
        vak_cols = ['pfizer-1', 'moderna-1', 'astra-1', 'sputnik-1', 'jnj']
        for i, col in enumerate(resistance_cols):
            df_idk[col] = 0
            df_idk.at[df_idk['vak']==vak_cols[i], col] = 1
        df.at[df_idk['chosen'], resistance_cols+['vac', 'R_date']] = df_idk[resistance_cols+['vac', 'R_date']].values
    
        # Generate random vakcinations
        df_vac_agg[vak_cols] = df_vac_agg[vak_cols].astype(int)
        df_vac_agg['past'] = df_vac_agg['date'] <= df_vac.date.max()
        df_vac_agg['date_i'] = np.arange(0,len(df_vac_agg)) - df_vac_agg['past'].sum()
        N_to_vax = int(df_vac_agg.loc[df_vac_agg.past==False][vak_cols].sum().sum())
        vax_ind = np.random.choice(df.loc[(df['Id']==0) & (df['vac']==0)].index, size=N_to_vax, replace=False)
    
        n_vac_future_arr = np.zeros(6, dtype=int)
        n_vac_future_arr[1:] = df_vac_agg.loc[df_vac_agg.past==False][vak_cols].sum().values.astype(int).cumsum()
        vac_arr = np.zeros((N_to_vax, 7), dtype=int)
        for n in range(5):
            vac_arr[n_vac_future_arr[n]:n_vac_future_arr[n+1],n] = 1    
            temp = df_vac_agg.loc[df_vac_agg.past==False][['date_i', vak_cols[n]]]
            r_dates = np.repeat(temp.date_i, temp[vak_cols[n]]).reset_index(drop=True).values
            np.random.shuffle(r_dates)
            vac_arr[n_vac_future_arr[n]:n_vac_future_arr[n+1],6] = r_dates
        vac_arr[:,5] = 1 # 'vac' column
        vac_arr[:,6] += (df_vac.date.max() - start_date).days +1 # add shift between sim start date and vaccination data
        df.at[vax_ind, resistance_cols+['vac', 'R_date']] = vac_arr   
        
        '''
        population = 5458827/100
        (df_vac_agg.set_index('date')/population).cumsum().plot(
            title="Prvé dávky",
            y=["pfizer-1", "moderna-1", "astra-1", "sputnik-1", "jnj"],
            kind="area",
            grid=True,
            ylabel='% z celej populácie',
            xlabel='Dátum'
                                          )
        #'''
        
        tempi = df.groupby(['R_date'])[resistance_cols].sum()
        population = 5458827/100
        (tempi/population).cumsum().plot(
            title="Prvé dávky",
            y=resistance_cols,
            kind="area",
            grid=True,
            ylabel='% z celej populácie',
            xlabel='Dátum'
                                          )
        zaock = df.loc[df['vek']>50].groupby(['R_date', 'lokalita_kod'])[resistance_cols].sum().unstack('lokalita_kod')
        zaock = zaock.fillna(0)
        zaock_full = zaock.copy()
        for col in zaock_full.columns:
            zaock_full[col].values[:] = 0
        
        for vaccine in resistance_cols:
            delay = df_vac_times.loc[vaccine]['vyhlaska_delay']
            zaock_full.at[zaock_full.index[delay:], vaccine] = zaock[vaccine].iloc[:-delay].values
        zaock_full = zaock_full.sum(level=1, axis=1).cumsum()
        vek_gr_cols = ['50-54','55-59','60-64','65-69','70-74','75-79','80-84','85-89','90-94','95-99','100-104','105-109','110-115']
        df_dem['nad_50'] = df_dem[vek_gr_cols].sum(axis=1)
        zaock_full[:] = zaock_full.values / df_dem.sort_values(['lokalita_kod'])['nad_50'].values * 100
        zaock_stupne = zaock_full.copy()
        x = zaock_stupne.values
        x = np.where(x<65, 0, x)
        x = np.where(x>=95, 4, x)
        x = np.where(x>=85, 3, x)
        x = np.where(x>=75, 2, x)
        x = np.where(x>=65, 1, x)
        zaock_stupne[:] = x
        times = fn.timeit(t=times, s='Vaccinations initialized', interval=[-1,-2]) 
        df['R_date'] = df['R_date'].fillna(0)
        
        
        
        '''
        fn.save_df(df, 'prep2_df.csv', path=params_dir, index=False, replace=True) 
        times = fn.timeit(t=times, s='Saved', interval=[-1,-2])   
        df = fn.load_df('prep2_df.csv', path=params_dir)
        times = fn.timeit(t=times, s='Loaded', interval=[-1,-2])
        #exit()
        ''' 
        # Enforce faster data types
        cols = df.columns.to_list()
        dtype_groups = df.dtypes.reset_index().groupby([0])['index'].unique().reset_index()
        object_cols = dtype_groups.loc[dtype_groups[0]=='object']['index'].values[0].tolist()
        dfo = df[object_cols].copy() #['vek_skup_5r', 'vek_skup_vak', 'lokalita']
        for item in object_cols:
            cols.remove(item)
        df = df[cols].astype('float32')
        times = fn.timeit(t=times, s='Data types enforced', interval=[-1,-2])
    
        resistance_cols = ['Rvp', 'Rvm', 'Rva', 'Rvs', 'Rvj']
        #res_inf_prot_cols = [col+'_inf_prot' for col in resistance_cols]
        i_states_cols = ['Id', 'Ih', 'Is', 'Ia']
        
        
        # Initialization states for Automat
        df_res = pd.DataFrame(columns=['day'])
        df_NAR = pd.DataFrame({'stupen':[1,1], 'stupen_calc':[1,1]}, index=[-2,-1])
        for i in range(1,8):
            df_NAR[str(i)] = 0
        df_NAR[str(4)] = df_dem.lokalita_kod.nunique()
        inci_okr_init = np.zeros((15,df_dem.lokalita_kod.nunique()))
        for i in range(14):
            temp_df = pd.DataFrame({'lokalita_kod':df_dem['lokalita_kod'].astype(float)},index=np.arange(0, df_dem['lokalita_kod'].nunique()))
            temp = df.loc[df['I_date']==-i].groupby('lokalita_kod')[i_states_cols].sum().sum(axis=1).reset_index()
            temp_df = fn.join_dfs(temp_df, temp, on='lokalita_kod', cols=['lokalita_kod', 0], add_on_prefix=False)
            inci_okr_init[-i,:] = temp_df[0].fillna(0).values
        df_okr = pd.DataFrame(data=inci_okr_init, columns=df_dem.lokalita_kod)
        '''
        fn.save_df(df_okr, 'prep_df_okr.csv', path=params_dir, index=False, replace=True) 
        times = fn.timeit(t=times, s='Saved', interval=[-1,-2])
        exit()
        df_okr = fn.load_df('prep_df_okr.csv', path=params_dir)
        '''
        df_okr.columns = df_dem.lokalita_kod
        times = fn.timeit(t=times, s='Automat initialized', interval=[-1,-2])
        times = fn.timeit(t=times, s='All ready', interval=[-1,0])
        res_inf_prot_max_cols = [col+'_inf_prot_max' for col in resistance_cols]
        res_hosp_prot_max_cols = [col+'_hosp_prot_max' for col in resistance_cols]
        max_inf_prots = df[res_inf_prot_max_cols].values
        max_hosp_prots = df[res_hosp_prot_max_cols].values
    
        # Sim params
        Sim_params = dict(
            days=days,
            df=df,
            df_res=df_res,
            df_NAR=df_NAR,
            df_okr=df_okr,
            test_ratio=test_ratio,
            new_inf_delay=new_inf_delay,
            max_inf_prots=max_inf_prots,
            max_hosp_prots=max_hosp_prots,
            R0s=R0s,
            max_inf_prots_in_time=max_inf_prots_in_time,
            max_hosp_prots_in_time=max_hosp_prots_in_time,
            post_inf_full_R=post_inf_full_R,
            R_post_inf=R_post_inf,
            auto_r0=auto_r0,
            df_dem=df_dem,
            #df_vak=df_vak,
            df_dis_times=df_dis_times,
            df_vac_times=df_vac_times,
            df_auto=df_auto,
            daily_imports=daily_imports,
            df_inf_ind_probs=df_inf_ind_probs,
            slabsi_automat=slabsi_automat,
            zaock_stupne=zaock_stupne,
            df_auto_new=df_auto_new,
            )
    
        # Simulation run
        df, df_res, df_NAR, df_okr, df_true_lvls, df_okr_inci = sim(**Sim_params)
        times = fn.timeit(t=times, s='Sim run', interval=[-1,-2])
        df_res = fn.join_dfs(df_res, df_R.DATUM.reset_index(), on='day', cols=['index','DATUM'])
        
        df_res['month_DATUM'] = df_res['day_DATUM'].dt.month
        df_res['deaths'] = 0
        df_res['deaths'].iloc[1:] = df_res['dead'].iloc[1:].values - df_res['dead'].iloc[:-1].values
        df_res['deaths_cum'] = df_res['deaths'].cumsum()
        df_res['future_hosp'] = df_res['new_future_hosp'].cumsum()
        df_res['tested_pos'] = np.nan
        df_res['tested_pos'].iloc[new_inf_delay:] = df_res['new_infections'].iloc[:-new_inf_delay].values/test_ratio
        for i in range(1, 1+new_inf_delay):
            df_res.at[i-1, 'tested_pos'] = df.loc[df['I_date']==-new_inf_delay-1+i][i_states_cols].sum().sum()/test_ratio
        
        #PLOTTING
        if to_plot:
            plt.rcParams["figure.dpi"] = 300
            '''
            #logged simulation params
            plt.title('Simulation')
            plt.plot(df_res['day'], df_res['inf_resistance'])
            plt.plot(df_res['day'], df_res['vaccinated'])
            plt.plot(df_res['day'], df_res['were_infected'])
            plt.grid(True)
            plt.ylim([0, len(df)])
            plt.ylabel('People [-]')
            plt.xlabel('Time [day]')
            plt.legend(['inf_resistance', 'vaccinated', 'were_infected'])
            plt.show()
            '''
    
            #plt.title('Simulation')
            fig, ax1 = plt.subplots()
            ax1.title.set_text(title)
            color = 'tab:red'
            ax1.set_xlabel('Time [date]')
            ax1.set_ylabel('People [-]')
            cols = ['new_infections', 'tested_pos', 'hosp', 'deaths_cum']
            leg_cols = ['actual infections', 'tested positive', 'hospitalized', 'deaths cumulative']
            for col in cols:
                ax1.plot(df_res['day_DATUM'], df_res[col])
            #'''   
            #sample_dates = [pd.to_datetime("2021-9-1"),pd.to_datetime("2021-10-9")]
            #sample_dates = [pd.to_datetime(df_vak[df_vak.date_i==ind_start]['date'].astype(str).values[0]),
            #                pd.to_datetime(df_vak[df_vak.date_i==ind_start+t_transition]['date'].astype(str).values[0])]
            sample_dates = [df_res.loc[df_res['day']==ind_start]['day_DATUM'].values[0],
                            df_res.loc[df_res['day']==ind_start+t_transition]['day_DATUM'].values[0],
                            pd.to_datetime('2021-08-16').to_numpy()]
            sample_dates = mdates.date2num(sample_dates)
            ymax = 5000
            plt.vlines(x=sample_dates, ymin=0, ymax=ymax, color = 'r')
            #plt.text(sample_dates[0], int(df_res['new_infections'].max()*0.4), 'Ind var 0%', rotation=90)
            #plt.text(sample_dates[1], int(df_res['new_infections'].max()*0.4), 'Ind var 100%', rotation=90)
            plt.text(sample_dates[0], int(ymax/1.6), 'Ind var 0%', rotation=90)
            plt.text(sample_dates[1], int(ymax/1.6), 'Ind var 100%', rotation=90)
            plt.text(sample_dates[2], int(ymax/3.5), 'Automat change', rotation=90)
            #plt.text(sample_dates[1], 4, 'Ind var 100%', rotation=90, verticalalignment='bottom')
            #'''
            ax1.set_xticks(df_res['day'])
            ax1.set_xticklabels(df_res['day_DATUM'], rotation='vertical', fontsize=8)
            ax1.set_ylim([-100, ymax])
            ax1.grid(True)
            ax1.legend(leg_cols)
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel('COVID Automat [black]')  # we already handled the x-label with ax1
            x = np.zeros(df_NAR.iloc[2:]['i'].values.shape[0]*2-1, dtype='int')
            x[::2] = df_NAR.iloc[2:]['i'].values
            x[1:-1:2] = x[2::2]-1
            y = np.zeros(x.shape)
            y[::2] = (df_NAR.iloc[2:][['1','2','3','4','5','6','7']]*np.array([1,2,3,4,5,6,7])).sum(axis=1)/79
            y[1:-1:2] = y[0:-2:2]
            ax2.plot(df_res.loc[x]['day_DATUM'], y, color='k')
            ax2.set_ylim([0,5])
            #ax2.legend(['levels'])
            fig.tight_layout()
            plt.show()
            times = fn.timeit(t=times, s='Plotted', interval=[-1,-2])
        return df_res, df_NAR, times, df, df_okr, df_true_lvls, df_okr_inci

    vak_list = ['50', '60', '70', '80']
    vak_choice = vak_list[2]
    ind_start_list = ['July', 'mid-July', 'August', 'September']
    ind_start_date = ind_start_list[1]
    inputs=[
            ['51','mid-July', 1],
            ['60','July', 0],
            ['50','July', 2],
            ['50','July', 1],
            ['70','August', 0],
            #['70','July'],
            #['70','September'],
            #['80','August'],
            #['70','September'],
            ]
    ress=[]
    for i, inputt in enumerate(inputs):
        df_res, df_NAR, times, df, df_okr, df_true_lvls, df_okr_inci = full_sim(times, vak_choice=inputt[0],
                                                                                 ind_start_date=inputt[1],
                                                                                 slabsi_automat=inputt[2],
                                                                                 to_plot=True)
        ress.append([df_res.copy(), df_NAR.copy()])
        break
    
    rem_per_pac = 0.34630574
    for i in range(len(ress)):
        temp = ress[i][0].groupby(['month_DATUM'])['new_future_hosp'].sum().reset_index()
        temp['rem'] = temp['new_future_hosp']*rem_per_pac
        print(inputs[i])
        #print(temp.iloc[1:4])
        print(temp.iloc[3:]['rem'].sum())


    today = pd.to_datetime("today")    
    i_today = df_res.loc[df_res['day_DATUM'].dt.date==today.date()]['day'].values[0]
    dochodkovy_vek = 62
    detsky_vek = 10
    prac_pop = [18, dochodkovy_vek]
    'detske_infections'
    'prac_infections'
    'doch_infections'
    df_res['I_date'] = df_res['day']
    df = fn.join_dfs(df, df_res, on='I_date', cols=['I_date', 'month_DATUM', 'day_DATUM'], add_on_prefix=False)
    print(df_res.groupby(['month_DATUM'])[['deaths','new_infections']].sum())
    print(df.loc[(df['I_date']>=i_today-17) & (df['I_date']<229-17) & (df['Id']==1)].groupby(['vek']).size())
    
    print('\nIs+Ih+Id v detskom veku podla mesiacov od dnes')
    print(df.loc[(df['vek']<detsky_vek) & (df['I_date']>=i_today) & \
                 ((df['Is']==1)|(df['Ih']==1)|(df['Id']==1))].groupby(['month_DATUM']).size())
    print('\nIs+Ih+Id v pracovnom veku podla mesiacov od dnes')
    print(df.loc[(df['vek']>prac_pop[0]) & (df['vek']<prac_pop[1]) & (df['I_date']>=i_today) & \
                 ((df['Is']==1)|(df['Ih']==1)|(df['Id']==1))].groupby(['month_DATUM']).size())
    print('\nIs+Ih+Id v dochodkovom veku podla mesiacov od dnes')
    print(df.loc[(df['vek']>dochodkovy_vek) & (df['I_date']>=i_today) & \
                 ((df['Is']==1)|(df['Ih']==1)|(df['Id']==1))].groupby(['month_DATUM']).size())
    '''
    df_dem = fn.load_df('vekove-skupiny-okresy_cut.xlsx', path=demografia_dir)
    df_1 = df_true_lvls.reset_index()
    df_1['day'] = np.arange(84,229,7)
    df_1 = fn.join_dfs(df_1, df_res, on='day', cols=['day','day_DATUM'], add_on_prefix=False)
    df_1 = df_1.set_index('day_DATUM')
    df_1 = df_1.drop(['index','day'], axis=1)
    df_1 = df_1.transpose().reset_index()
    df_1['lokalita_kod'] = df_1['index']
    df_1 = fn.join_dfs(df_1, df_dem, on='lokalita_kod', cols=['lokalita_kod','lokalita'], add_on_prefix=False)
    df_1 = df_1.drop(['index'], axis=1)
    fn.save_df(df_1, 'stav_automatu.xlsx', path=params_dir, index=False, replace=True)
    
    
    df_2 = df.loc[df['I_date']>106][['vek','lokalita_kod','Ia','Is','Ih','Id','I_date','day_DATUM']].reset_index(drop=True).copy()
    df_2 = fn.join_dfs(df_2, df_dem, on='lokalita_kod', cols=['lokalita_kod','lokalita'], add_on_prefix=False)
    df_2 = df_2.drop(['I_date'], axis=1)
    fn.save_df(df_2, 'infikovany.xlsx', path=params_dir, index=False, replace=True)
    '''
    #fn.save_df(df_res, 'midJuly-51-update.xlsx', path=params_dir, index=False, replace=True)
    winsound.Beep(frequency, duration)
    
    
    '''
    for i in range(len(ress)):
        temp = ress[i][0][['day_DATUM','hosp']]
        fn.save_df(temp, f'max_vak-{inputs[i][0]}_ind_comes-{inputs[i][1]}.xlsx', path=params_dir, index=False, replace=True) 
    
    df_res, df_NAR, times = full_sim(times, vak_choice='80',
                                 ind_start_date='September',
                                 to_plot=True)

    #vaccine infection and hospitalization protection
    days = df_vac_times['protection_delay'].max() + 11
    x = np.arange(0,days)
    df_vac = pd.DataFrame({'R_date':x[::-1], 'vac':1}, index = x)   
    for col in i_states_cols:
        df_vac[col] = 0
    for col in resistance_cols:
        df_vac[col] = 0
        df_vac[col+'_inf_prot_max'] = df_vac_times.loc[col]['inf_protection']
        df_vac[col+'_hosp_prot_max'] = df_vac_times.loc[col]['hosp_protection'] 
    df_vac = get_Rvac(df_vac, 95)

    plt.title('Vaccine infection protection')
    for col in resistance_cols:
        plt.plot(df_vac.index, df_vac[col+'_inf_prot'])
    plt.grid(True)
    plt.ylabel('Infection protection [-]')
    plt.xlabel('Time since first dose [day]')
    plt.legend(resistance_cols)
    plt.show()

    plt.title('Vaccine hospitalisation protection')
    for col in resistance_cols:
        plt.plot(df_vac.index, df_vac[col+'_hosp_prot'])
    plt.grid(True)
    plt.ylabel('Hospitalisation protection [-]')
    plt.xlabel('Time since first dose [day]')
    plt.legend(resistance_cols)
    plt.show()
    # Disease trajectory chances per age groups
    for col in inf_prob_cols:
        plt.plot(df_inf_probs.vek, df_inf_probs[col])
    plt.grid(True)
    plt.ylabel('Disease trajectory chance [-]')
    plt.xlabel('Age [years]')
    plt.legend(inf_prob_cols)
    plt.show()
    
    # Number of infected people per infectious compared to slovak contact tracing data
    arr = np.zeros(718)
    df_nak_probs = pd.DataFrame({'nakazeny':1, 'nakazil':arr, 'nakazil_gen':np.random.geometric(0.839766, 718)-1}, index = np.arange(0,arr.shape[0]))
    df_nak_probs.at[[0,1,2], 'nakazil'] = 11
    df_nak_probs.at[[3], 'nakazil'] = 10
    df_nak_probs.at[[4,5], 'nakazil'] = 7
    df_nak_probs.at[[6], 'nakazil'] = 5
    df_nak_probs.at[[7], 'nakazil'] = 3
    df_nak_probs.at[np.arange(8,8+13), 'nakazil'] = 2
    df_nak_probs.at[np.arange(8+13, 8+13+46), 'nakazil'] = 1
    temp = df_nak_probs['nakazil'].value_counts().reset_index().sort_values('index')
    plt.plot(temp['index'], temp['nakazil']/temp['nakazil'].sum())
    temp = df_nak_probs['nakazil_gen'].value_counts().reset_index().sort_values('index')
    plt.plot(temp['index'], temp['nakazil_gen']/temp['nakazil_gen'].sum())
    plt.grid(True)
    plt.ylabel('probability [-]')
    plt.xlabel('N of infected people [-]')
    plt.legend(['contact tracing', 'approximation function'])
    plt.show()
    #'''
    
    times = fn.timeit(t=times, s='Plotted', interval=[-1,-2])

    
    #times = fn.timeit(t=times, s='pracovnici naparovany', interval=[-1,-2])
    times = fn.timeit(t=times, s='Full runtime', interval=[-1,0])






    
    
if False:
    plt.title('Simulation')
    for col in df_umrt.columns[1:-2]:
        plt.plot(df_umrt['DATUM'], df_umrt[col])
    plt.grid(True)
    plt.ylabel('People [-]')
    plt.xlabel('Time [day]')
    plt.legend(df_umrt.columns[1:])
    plt.show()
    
    plt.plot(df_umrt['DATUM'], df_umrt['spolu'])
    plt.plot(df_umrt['DATUM'], fn.seven_davg(df_umrt['spolu'].values))
    plt.grid(True)
    plt.ylabel('spolu umrtia[-]')
    plt.xlabel('Time [date]')
    plt.legend(['OG', '7day avg (not trailing)'])
    plt.show()   
    
    
    plt.plot(df_hosp['DATUM'], df_hosp['spolu'])
    plt.plot(df_hosp['DATUM'], fn.seven_davg(df_hosp['spolu'].values))
    plt.grid(True)
    plt.ylabel('spolu hospitalizacie[-]')
    plt.xlabel('Time [date]')
    plt.legend(['OG', '7day avg (not trailing)'])
    plt.show()   
    
    'PCR_N', 'PCR_pos', 'PCR_neg', 'PCR_TPR', 'AG_N'
    plt.plot(df_umrt['DATUM'], fn.seven_davg(df_umrt['spolu'].values)/fn.seven_davg(df_umrt['spolu'].fillna(0).values).max())
    plt.plot(df_hosp['DATUM'], fn.seven_davg(df_hosp['spolu'].values)/fn.seven_davg(df_hosp['spolu'].fillna(0).values).max())
    plt.plot(df_test['DATUM'], fn.seven_davg(df_test['PCR_pos'].values)/fn.seven_davg(df_test['PCR_pos'].fillna(0).values).max())
    plt.plot(df_test['DATUM'], fn.seven_davg(df_test['PCR_TPR'].values)/fn.seven_davg(df_test['PCR_TPR'].fillna(0).values).max())
    #plt.plot(df_test['DATUM'], fn.seven_davg(df_test['AG_pos'].values)/fn.seven_davg(df_test['AG_pos'].fillna(0).values).max())
    #plt.plot(df_test['DATUM'], fn.seven_davg(df_test['AG_TPR'].values)/fn.seven_davg(df_test['AG_TPR'].fillna(0).values).max())
    plt.grid(True)
    plt.ylabel('Normalizovany pocet [-]')
    plt.xlabel('Time [date]')
    plt.legend(['umrtia', 'hospitalizacie', 'PCR_pos', 'PCR_TPR', 'AG_pos', 'AG_TPR'])
    plt.show()   
    
    plt.plot(df_hosp['DATUM'], fn.seven_davg(df_hosp['spolu'].values)/fn.seven_davg(df_hosp['spolu'].fillna(0).values).max()/\
             (df_umrt['spolu'].values)/fn.seven_davg(df_umrt['spolu'].fillna(0).values).max())
    plt.grid(True)
    plt.ylabel('Normalizovany pocet [-]')
    plt.xlabel('Time [date]')
    plt.legend(['umrtia', 'hospitalizacie'])
    plt.show()   
    
    
    #pd.Series(df_umrt.columns[1:-2].values).str.split('-').apply(lambda x: int((float(x[0]) + float(x[1]))/2))
    
    def groupby_vek_cols(df, vek_cols):
        df_prob = df[vek_cols].sum().reset_index()
        df_prob.columns = ['vek','N']
        temp = df_prob['vek'].str.split('-').str
        df_prob['vek'] = (temp[0].astype(int) + temp[1].astype(int))/2
        df_prob = df_prob.append({'vek':115, 'N':0}, ignore_index=True)
        df_prob['prob'] = df_prob['N']/df_prob['N'].sum()
        return df_prob
    
    
    pop_prob = groupby_vek_cols(df_dem, df_dem.columns[3:])
    hosp_prob = groupby_vek_cols(df_hosp, df_hosp.columns[1:-2])
    umrt_prob = groupby_vek_cols(df_umrt, df_umrt.columns[1:-2])
    
    
    ages = np.arange(0,116)
    df_inf_probs = pd.DataFrame({'vek':ages}, index = ages) 
    df_inf_probs['pop_prob'] = interpolation(pop_prob['vek'], pop_prob['prob'], ages)
    df_inf_probs['hosp_prob'] = interpolation(hosp_prob['vek'], hosp_prob['prob'], ages)
    df_inf_probs['umrt_prob'] = interpolation(umrt_prob['vek'], umrt_prob['prob'], ages)
    df_inf_probs['pop_prob'] = df_inf_probs['pop_prob']/df_inf_probs['pop_prob'].sum()
    df_inf_probs['hosp_prob'] = df_inf_probs['hosp_prob']/df_inf_probs['hosp_prob'].sum()
    df_inf_probs['umrt_prob'] = df_inf_probs['umrt_prob']/df_inf_probs['umrt_prob'].sum()
    
    
    for col in df_inf_probs.columns[5:]:
        plt.plot(df_inf_probs['vek'], df_inf_probs[col])
    plt.grid(True)
    plt.ylabel('Normalizovany pocet [-]')
    plt.xlabel('age [rok]')
    plt.legend(df_inf_probs.columns[1:])
    plt.show()   
    
    
    def norm(idk):
        idk = pd.Series(idk)
        return idk/idk.max()
    
    plt.plot(df_inf_probs['vek'], norm(fn.seven_davg((df_inf_probs['umrt_prob']/df_inf_probs['pop_prob']).values)))
    plt.plot(df_inf_probs['vek'], norm(fn.seven_davg((df_inf_probs['hosp_prob']/df_inf_probs['pop_prob']).values)))
    plt.plot(df_inf_probs['vek'], norm(df_inf_probs['Id_prob']))
    plt.plot(df_inf_probs['vek'], norm(df_inf_probs['Ih_prob']))
    plt.grid(True)
    plt.ylabel('Normalizovany pocet [-]')
    plt.xlabel('age [rok]')
    plt.legend(['umrt_prob','hosp_prob', 'Id_prob','Ih_prob'])
    plt.show()   
    
    plt.grid(True)
    plt.ylabel('Normalizovany pocet [-]')
    plt.xlabel('age [rok]')
    plt.legend(['Id_prob','Ih_prob'])
    plt.show()   
    
    
    
    I_arr = df[['Ia_prob', 'Is_prob', 'Ih_prob', 'Id_prob']].values
    inf_rng = np.random.uniform(size=I_arr.shape[0])
    I_ind_arr = (I_arr > inf_rng[:,np.newaxis]).argmax(axis=1)
    pop_i_prob = np.zeros((I_arr.shape[0],4), dtype='float32')
    for j in range(4):
        pop_i_prob[:,j] = (I_ind_arr == j).astype('float32')
    pop_i_prob = pop_i_prob.sum(axis=0)/pop_i_prob.sum()
    
    
    
  
    arr = fn.seven_davg(df_test['pos'].values)
    arr = fn.seven_davg(arr[1:]/arr[:-1])
    #print(arr)
    arr = fn.seven_davg(df_res['new_infections'].values)
    arr = fn.seven_davg(arr[1:]/arr[:-1])
    print(arr[-1])








