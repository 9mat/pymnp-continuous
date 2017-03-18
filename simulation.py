# %%
import pandas as pd
import numpy as np
import sys
import json

setting_file = './settings.json'
    
settings = {}
with open(setting_file, 'r') as f:
    settings = json.load(f)

def read_settings(attr, default_value=None):
    return settings[attr] if attr in settings else default_value


dta_input_path = read_settings('dta_input_path')
solution_path  = read_settings('solution_path')
bstr_prefix    = read_settings('bstr_prefix', './bstr')
save_solution  = read_settings('save_solution', True)

# Fixed effects
utilfe = read_settings('utilfe', False)
expdfe = read_settings('expdfe', False)

# home_dir = '/home/dhlong/UbuntuData/Dropbox/PhD Study/research/mnp_demand_gas_ethanol_dhl/continous/'
# #home_dir = '/hpctmp2/dhlong/continous/'
# dta_input_path = home_dir + '../individual_data_wide.dta'
# solution_path = home_dir + 'solution.npy'

with open(dta_input_path, 'rb') as fi:
    df = pd.read_stata(fi, encoding='latin-1')

# convert indexing variables to int
df.treattype = df.treattype.astype(int)
df.choice = df.choice.astype(int)
df.consumerid = df.consumerid.astype(int)
df.stationid = df.stationid.astype(int)

# old coding: 2 = midgrade gasoline, 3 = ethanol
# new coding: 2 = ethanol, 3 = midgrad gasoline
choice = df.choice.as_matrix()
df.loc[choice==3, 'choice']=2
df.loc[choice==2, 'choice']=3

# drop RJ, drop midgrade ethanol and treatment 3 and 4
df = df[df.dv_rj==0]
df = df[df.choice < 3]
df = df.loc[df.treattype < 3]

nobs = len(df)
prices = df[['pg_km_adj', 'pe_km_adj']].as_matrix()

nsims = 200
lelas = -0.2
prob = 0.2
lmu = 0
lsigma = -0.4

simid = 1

if len(sys.argv) > 1:
	nsims = int(sys.argv[1])
if len(sys.argv) > 2:
	lelas = float(sys.argv[2])
if len(sys.argv) > 3:
	simid = int(sys.argv[3])


import random

for i in range(nsims):
	nu = np.random.gumbel(size=prices.shape)
	eta = np.random.normal(size=(prices.shape[0],))

	mu = np.exp(lmu)
	sigma = np.exp(lsigma)

	elas = np.exp(lelas)

	theta = np.exp(sigma*eta + 4.5).reshape(-1,1)
	quant = theta*prices**(-elas)
	payments = quant*prices
	utilquant = theta*prices**(1-elas)/(elas-1)
	util = utilquant + nu*mu

	choice = np.argmax(util,axis=1)

	observed_payments = payments[np.arange(payments.shape[0]), choice]

	quant_fixed = 50/prices
	util_fixed = theta**(1/elas)*quant_fixed**(1-1/elas)/(1-1/elas) -50 + nu*mu
	choice_fixed = np.argmax(util_fixed,axis=1)

	fixed_subsample = random.sample(range(nobs), int(prob*nobs))

	choice[fixed_subsample] = choice_fixed[fixed_subsample]
	observed_payments[fixed_subsample] = 50

	df['choice'] = choice + 1
	df['value_total'] = observed_payments

	df.to_stata('../../simdata/set{}/individual_wide_sim{:04}.dta'.format(simid, i), encoding='latin-1')