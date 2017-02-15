# %% Cell 1
    
import pandas as pd
import numpy as np
import theano.tensor as T
import theano
import theano.tensor.extra_ops as op
import pyipopt
import os
import sys
import pickle
import json

setting_file = './settings.json'
with open(setting_file, 'r') as f:
    settings = json.load(f)
    dta_input_path = settings['dta_input_path']
    solution_path = settings['solution_path']
    bstr_prefix = settings['bstr_prefix']

# home_dir = '/home/dhlong/UbuntuData/Dropbox/PhD Study/research/mnp_demand_gas_ethanol_dhl/continous/'
# #home_dir = '/hpctmp2/dhlong/continous/'
# dta_input_path = home_dir + '../individual_data_wide.dta'
# solution_path = home_dir + 'solution.npy'

with open(dta_input_path, 'rb') as fi:
    df = pd.read_stata(fi)

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
df = df[df.choice < 4]
df = df.loc[df.treattype < 3]

# df['treat1'] = df.treattype == 1
# df['ntreat1'] = df.groupby(['date', 'stationid']).treat1.transform(sum)
# df = df[df.ntreat1 > 0]

# re-index station in running order (0 to nstation-1)
unique_station, station_reverse_id = np.unique(df.stationid.as_matrix(), return_inverse=True)
nstation = len(unique_station)
new_stationid = np.arange(nstation)
df.stationid = new_stationid[station_reverse_id]

# # generate stations dummies
# station_dummies = pd.get_dummies(df['stationid'], prefix='dv_station')
# df[station_dummies.columns[1:]] = station_dummies[station_dummies.columns[1:]]

# generate day of week dummies
dow_dummies = pd.get_dummies(df['date'].dt.dayofweek, prefix='dv_dow')
df[dow_dummies.columns[1:]] = dow_dummies[dow_dummies.columns[1:]]

# treatment dummies
for elem in df['treattype'].unique():
    df.loc[:,'treat' + str(elem)] = df['treattype'] == elem

# choice dummies
for elem in df['choice'].unique():
    df.loc[:,'choice' + str(elem)] = df['choice'] == elem

# inputfile = './data_new_volume.csv'
# df = pd.read_csv(inputfile)

# df = df.sample(2000)

# impute missing prices
df['pgmidgrade_km_adj'].fillna(value=1e9, inplace=True)
df['pemidgrade_km_adj'].fillna(value=1e9, inplace=True)

# generate new variables
df.loc[:,'const'] = 1

df.loc[:,'treat1_topusage'] = df['treat1']*df['dv_usageveh_p75p100']
df.loc[:,'treat2_topusage'] = df['treat2']*df['dv_usageveh_p75p100']

df.loc[:,'treat1_college'] = df['treat1']*df['dv_somecollege']
df.loc[:,'treat2_college'] = df['treat2']*df['dv_somecollege']

df.loc[:,'ltank'] = np.log(df['car_tank'])


price_labels = ['pg_km_adj', 'pe_km_adj', 'pgmidgrade_km_adj', 'pemidgrade_km_adj']
price_labels = price_labels[:max(df['choice'])]
value_labels = ['value_total']

# all
# Xexpd_labels = ['dv_ctb', 'dv_bh', 'dv_rec', 'dv_female', 'dv_age_25to40y', 'dv_age_morethan65y', 'dv_somesecondary', 'dv_somecollege', 'dv_carpriceadj_p75p100', 'dv_usageveh_p75p100', 'stationvisit_avgcarprice_adj', 'const'] #+ list(station_dummies.columns[1:])
# Xutil_labels = ['dv_ctb', 'dv_bh', 'dv_rec', 'dv_female', 'dv_age_25to40y', 'dv_age_morethan65y', 'dv_somesecondary', 'dv_somecollege', 'dv_carpriceadj_p75p100', 'dv_usageveh_p75p100', 'stationvisit_avgcarprice_adj', 'const'] #+ list(station_dummies.columns[1:])
Xexpd_labels = ['ltank', 'dv_female', 'dv_age_25to40y', 'dv_age_morethan65y', 'dv_somesecondary', 'dv_somecollege', 'dv_carpriceadj_p75p100', 'dv_usageveh_p75p100', 'stationvisit_avgcarprice_adj', 'dv_dow_1', 'dv_dow_2', 'dv_dow_3', 'dv_dow_4', 'dv_dow_5', 'dv_dow_6', 'dv_start_0901_1200',  'dv_start_1201_on', 'const'] #+ list(station_dummies.columns[1:])
Xutil_labels = ['ltank', 'dv_female', 'dv_age_25to40y', 'dv_age_morethan65y', 'dv_somesecondary', 'dv_somecollege', 'dv_carpriceadj_p75p100', 'dv_usageveh_p75p100', 'stationvisit_avgcarprice_adj', 'dv_dow_1', 'dv_dow_2', 'dv_dow_3', 'dv_dow_4', 'dv_dow_5', 'dv_dow_6', 'dv_start_0901_1200',  'dv_start_1201_on'] #+ list(station_dummies.columns[1:])

# usage
#Xexpd_labels = ['choice2', 'choice3', 'dv_ctb', 'dv_bh', 'dv_rec', 'dv_female', 'dv_age_25to40y', 'dv_age_morethan65y', 'dv_somesecondary', 'dv_somecollege', 'dv_carpriceadj_p75p100', 'stationvisit_avgcarprice_adj', 'const']
#Xutil_labels = ['dv_ctb', 'dv_bh', 'dv_rec', 'dv_female', 'dv_age_25to40y', 'dv_age_morethan65y', 'dv_somesecondary', 'dv_somecollege', 'dv_carpriceadj_p75p100', 'stationvisit_avgcarprice_adj', 'const']

# car price
#Xexpd_labels = ['dv_ctb', 'dv_bh', 'dv_rec', 'dv_female', 'dv_age_25to40y', 'dv_age_morethan65y', 'dv_somesecondary', 'dv_somecollege', 'dv_usageveh_p75p100', 'stationvisit_avgcarprice_adj', 'const']
#Xutil_labels = ['dv_ctb', 'dv_bh', 'dv_rec', 'dv_female', 'dv_age_25to40y', 'dv_age_morethan65y', 'dv_somesecondary', 'dv_somecollege', 'dv_usageveh_p75p100', 'stationvisit_avgcarprice_adj', 'const']

Xlelas_labels = ['const', 'treat1', 'treat2']
# Xlelas_labels = ['const', 'treat1', 'treat2', 'treat3', 'treat4']
# Xlelas_labels = ['const', 'treat1', 'treat2', 'dv_usageveh_p75p100', 'treat1_topusage', 'treat2_topusage']
# Xlelas_labels = ['const', 'treat1', 'treat2', 'dv_somecollege', 'treat1_college', 'treat2_college']
# Xlelas_labels = ['const', 'treat1', 'treat2', 'dv_usageveh_p75p100', 'dv_somecollege', 'treat1_topusage', 'treat2_topusage', 'treat1_college', 'treat2_college']
Xlsigma_labels = ['const']
Xlmu_labels = ['const']

floatX = 'float64'
intX = 'int32'

# find choice-station that no consumers choose --> unidentified fixed effects
df_nchoice = df[['stationid','choice', 'const']].groupby(['stationid','choice']).sum()
idf_fe = theano.shared((1-np.isnan(df_nchoice.unstack().as_matrix())).astype(floatX), 'idf_fe')

# prepare shared variables to pass to theano
choice = theano.shared((df.loc[:, 'choice'].as_matrix()-1).astype(intX), 'choice')
price  = theano.shared(df.loc[:, price_labels].as_matrix().astype(floatX), 'price')
value  = theano.shared(df.loc[:, value_labels].as_matrix().astype(floatX), 'value', broadcastable=(False, True))
Xexpd  = theano.shared(df.loc[:, Xexpd_labels].as_matrix().astype(floatX), 'Xexpd')
Xutil  = theano.shared(df.loc[:, Xutil_labels].as_matrix().astype(floatX), 'Xutil')
stationid  = theano.shared(df.loc[:, 'stationid'].as_matrix().astype(intX), 'stationid')

Xlelas = theano.shared(df.loc[:, Xlelas_labels].as_matrix().astype(floatX), 'Xlelas')
Xlmu   = theano.shared(df.loc[:, Xlmu_labels].as_matrix().astype(floatX), 'Xlmu')
Xlsigma = theano.shared(df.loc[:, Xlsigma_labels].as_matrix().astype(floatX), 'Xlsigma')

# dimensionality
nobs = len(df)
nchoice = len(price_labels)
nXexpd = len(Xexpd_labels)
nXutil = len(Xutil_labels)

nXlelas = len(Xlelas_labels)
nXlsigma = len(Xlsigma_labels)
nXlmu = len(Xlmu_labels)

dvchoice = T.eq(choice.reshape((-1,1)), np.arange(nchoice, dtype=int).reshape((1,-1)))
chosenprice = T.sum(price*dvchoice,axis=1,keepdims=True)
convenience_expend = 50.0
dvconvenience = T.abs_(op.squeeze(value)-convenience_expend) < 1e-3
n_convenience = dvconvenience.sum()
convenience = dvconvenience.nonzero()
inconvenience = (~dvconvenience).nonzero()
dvchoicef = dvchoice.astype(floatX)

# Settings
utilfe = False
expdfe = False

ntheta = (nXlelas + nXlsigma + nXlmu + nXexpd + 
    (nchoice-1)*nXutil + 
    (nstation-1 if expdfe else 0) + # expenditure station fixed effects
    ((nchoice-1)*(nstation) if utilfe else 0) + # utility station fixed effects
    (nchoice-1) + # alpha_j -- expenditure product fixed effect
    1) # probability of fixed payment

def getparams(theta, expdfe=False, utilfe=False):
    offset = 0
    gammalelas = theta[offset:offset+nXlelas].reshape((nXlelas, 1))
    
    offset += nXlelas
    gammalsigma = theta[offset:offset+nXlsigma].reshape((nXlsigma, 1))
    
    offset += nXlsigma
    gammalmu = theta[offset:offset+nXlmu].reshape((nXlmu, 1))
    
    offset += nXlmu
    betaexpd = theta[offset:offset+nXexpd].reshape((nXexpd, 1))
    
    offset += nXexpd
    betautil = theta[offset:offset+(nchoice-1)*nXutil].reshape((nXutil, nchoice-1))
        
    offset += (nchoice-1)*nXutil
    ltpconve = theta[offset]
    
    offset += 1
    alphaexpend = theta[offset:offset+nchoice-1]
    
    offset += nchoice-1

    betaexpdfe = None
    if expdfe:
        betaexpdfe = theta[offset:offset+(nstation-1)].reshape((-1,1))
        offset += (nstation-1)

    betautilfe = None
    if utilfe:
        betautilfe = theta[offset:offset+(nchoice-1)*(nstation)].reshape((-1,nchoice-1))
        offset += (nstation-1)*(nstation)
    
    return gammalelas, gammalsigma, gammalmu, betaexpd, betautil, ltpconve, alphaexpend, betaexpdfe, betautilfe

def logsumexp(x,axis,w=idf_fe):
    maxx = T.max(x,axis=axis,keepdims=True)
    if w is None or (not utilfe):
        return op.squeeze(maxx) + T.log(T.sum(T.exp(x-maxx),axis=axis))
    return op.squeeze(maxx) + T.log(T.sum(T.exp(x-maxx)*w[stationid],axis=axis))

def logsumexp2(x,y):
    m = T.maximum(x,y)
    return m + T.log(T.exp(x-m) + T.exp(y-m))

# theta = T.vector('theta')
theta = T.vector('theta64', dtype=floatX)

# def build_nlogl(theta):
gammalelas, gammalsigma, gammalmu, betaexpd, betautil, ltpconve, alphaexpend, betaexpdfe, betautilfe = getparams(theta, expdfe, utilfe)

if utilfe:
    betautilfe = T.concatenate([T.zeros((nstation,1)), betautilfe], axis=1) - 1e9*(1-idf_fe)
#     betautilfe = T.concatenate([T.zeros((nstation,1)), betautilfe], axis=1)
#     betautilfe = T.concatenate([T.zeros((1,nchoice)), betautilfe], axis=0)
    
if expdfe:
    betaexpdfe = T.concatenate([[[0]], betaexpdfe], axis=0)
    
alphaexpend = T.concatenate([T.zeros(1,), alphaexpend],axis=0)


pconve   = T.nnet.sigmoid(ltpconve)
mu       = T.exp(T.dot(Xlmu, gammalmu))
elas     = T.exp(T.dot(Xlelas, gammalelas))
lsigma   = T.dot(Xlsigma,gammalsigma)
rho      = elas - 1
elasdrho = elas/rho

###################################################################
# Model - consumer choice under flexible payment preference
###################################################################

lconvenience_expend = np.log(convenience_expend)

alphachoice = alphaexpend[choice].dimshuffle([0,'x'])

eta = T.log(value) + rho*T.log(chosenprice) - T.dot(Xexpd,betaexpd) - alphachoice - (betaexpdfe[stationid,:] if expdfe else 0)
lexpend = T.log(value) - rho*(T.log(price) - T.log(chosenprice)) + alphaexpend - alphachoice

utilhete = T.concatenate([T.zeros((Xutil.shape[0],1)), T.dot(Xutil, betautil)],axis=1) + (betautilfe[stationid,:] if utilfe else 0) 
utilquant = T.exp(lexpend)/rho

util0 = T.minimum((utilquant + utilhete)/mu, 1e9)
lprobchoice0 = T.sum(util0*dvchoice,axis=1) - logsumexp(util0,1)

lprobchoice = lprobchoice0 + np.log(1-pconve)
lpdfeta = eta*eta/(2*T.exp(2*lsigma)) + lsigma + np.log(2*np.pi)/2
ll1 = -lprobchoice + op.squeeze(lpdfeta)

# Integration to calcualte choice probability
# Hermite Gaussian quadrature
# http://keisan.casio.com/exec/system/1329114617
# $$\int_{-\infty}^{\infty} e^{-x^2}f(x)dx \approx \sum w_i f(x_i)$$

abscissa = np.array([-3.88972489786978000, -3.02063702512089000, -2.27950708050106000, -1.59768263515260000, -0.94778839124016300, -0.31424037625435900, 0.31424037625435900, 0.94778839124016300, 
                   1.59768263515260000, 2.27950708050106000, 3.02063702512089000, 3.88972489786978000
                  ]).reshape(-1,1,1).astype(floatX)

weight = np.array([0.00000026585516844, 0.00008573687043588, 0.00390539058462906, 0.05160798561588390, 
                     0.26049231026416100, 0.57013523626248000, 0.57013523626248000, 0.26049231026416100, 
                     0.05160798561588390, 0.00390539058462906, 0.00008573687043588, 0.00000026585516844
                     ]).reshape(-1,1,1).astype(floatX)



###################################################################
# Model - consumer choice under fixed-payment preference
###################################################################

eta_i = T.exp(lsigma)*abscissa
lexpend_i = lexpend - eta + eta_i
utilconve_i = (T.exp((lexpend_i-lconvenience_expend)/elas)/(1-1/elas) - 1)*convenience_expend
utilb = T.minimum((utilconve_i + utilhete)/mu, 1e9)
lprobchoice_i = T.sum(utilb*dvchoicef,axis=2) - logsumexp(utilb,2)
ll = lprobchoice_i + np.log(weight)[:,:,0] - np.log(2*np.pi)/2
ll2 = -logsumexp(ll,0,None) - T.log(pconve)

nlogl_v = ll1[inconvenience].sum() + logsumexp2(ll1, ll2)[convenience].sum()

#### end of model #################################################

nlogl_g = T.grad(nlogl_v, theta)
nlogl_h = theano.gradient.hessian(nlogl_v, theta)

floatXnp = 'float64'
type_conversion = lambda f: lambda x: f(x.astype(floatX)).astype(floatXnp)

eval_f = type_conversion(theano.function([theta], nlogl_v))
eval_g = type_conversion(theano.function([theta], nlogl_g, allow_input_downcast=True))
eval_h = type_conversion(theano.function([theta], nlogl_h, allow_input_downcast=True))


if os.path.isfile(solution_path):
    theta000 = np.load(solution_path)
else:
    theta000 = np.zeros(ntheta)
    theta000[0] = 0.1
    theta000 = theta000.astype(floatXnp)

#Minimize negative log-likelihood
pyipopt.set_loglevel(1)
thetahat , _, _, _, _, fval = pyipopt.fmin_unconstrained(
    type_conversion(eval_f),
    theta000,
    fprime=type_conversion(eval_g),
    fhess=eval_h,)

#np.save(solution_path, thetahat)

def print_row1(lbl, hat, se, t): 
    formatstr = '%30s%10.3f%10.3f%10.3f'
    print(formatstr % (lbl, hat, se, t) )

def print_row2(lbl, hat, se, t):
    star = np.sum(np.abs(t) > [1.65, 1.96, 2.58]) 
    print('{:>30}{:10.3f}{:<3}'.format(lbl, float(hat), '*'*star) )
    print('{:>40}]'.format('[{:.3f}'.format(float(se))) )
    
def print_row3(lbl, hat, se, t): 
    star = np.sum(np.abs(t) > [1.65, 1.96, 2.58]) 
    print('{},="{:.3f}",{}'.format(lbl, float(hat), '*'*star) )
    print(',="[{:.3f}]"'.format(float(se)))
    
def print_results(thetahat, print_row=print_row2):
    covhat = np.linalg.pinv(eval_h(thetahat.astype(floatX)))
    sehat = np.sqrt(np.diagonal(covhat))
    t = thetahat/sehat

    gammalelashat, gammalsigmahat, gammalmuhat, betaexpdhat, betautilhat, bhat, alphahat,_,_ = getparams(thetahat)
    gammalelasse, gammalsigmase, gammalmuse, betaexpdse, betautilse, bse, alphase, _, _ = getparams(sehat)
    gammalelast, gammalsigmat, gammalmut, betaexpdt, betautilt, bt, alphat, _, _ = getparams(t)

    print('-'*60)

    print(' \n*** ln(elas) equation')
    for i in range(nXlelas):
        print_row(Xlelas_labels[i], gammalelashat[i], gammalelasse[i], gammalelast[i])
    print('-'*60)

    print(' \n*** ln(sigma) equation')
    for i in range(nXlsigma):
        print_row(Xlsigma_labels[i], gammalsigmahat[i], gammalsigmase[i], gammalsigmat[i])
    print('-'*60)

    print(' \n*** ln(mu) equation')
    for i in range(nXlmu):
        print_row(Xlmu_labels[i], gammalmuhat[i], gammalmuse[i], gammalmut[i])
    print('-'*60)

    print(' \n*** Expenditure equation')
    for i in range(nchoice-1):
        print_row('alpha_' + str(i), alphahat[i], alphase[i], alphat[i])

    for i in range(nXexpd):
        print_row(Xexpd_labels[i], betaexpdhat[i], betaexpdse[i], betaexpdt[i])
    print( '-'*60)

    print(' \n*** Discrete choice equation')
    for j in range(nchoice-1):
        print('-------- choice', j+1, '------------------------------------------')
        for i in range(nXutil):
            print_row(Xutil_labels[i], betautilhat[i][j], betautilse[i][j], betautilt[i][j])
    print('-'*60)

    print(' \n*** logit prob convenience')
    print_row('const', bhat, bse, bt) 
    print('-'*60)
    
    dtreat = gammalelas[1,0] - gammalelas[2,0]
    grad_dtreat = T.grad(dtreat, theta)
    grad_value = theano.function([theta], grad_dtreat)(thetahat.astype(floatX))
    se_dtreat = np.sqrt(grad_value.dot(covhat).dot(grad_value))
    dtreat_hat = theano.function([theta], dtreat)(thetahat.astype(floatX))
    print(dtreat_hat)
    print(se_dtreat)
    print(dtreat_hat/se_dtreat)
    
print_results(thetahat)

from sklearn.utils import resample

def bootstrap_sample_by_station(df, resampled_stationid, nstation=nstation):
    assert(len(df['stationid'].unique()) == nstation)
    assert(df['stationid'].min()==0)
    assert(df['stationid'].max()==nstation-1)
    
    resampled_df = pd.DataFrame()
    for i in range(len(resampled_stationid)):
        df2 = df[df['stationid']==resampled_stationid[i]].copy()
        df2['stationid'] = i
        resampled_df = resampled_df.append(df2)
        
    return resampled_df.reset_index()

def set_shared_value(df):
    choice.set_value((df.loc[:, 'choice'].as_matrix()-1).astype(intX))
    price.set_value(df.loc[:, price_labels].as_matrix().astype(floatX))
    value.set_value(df.loc[:, value_labels].as_matrix().astype(floatX))
    Xexpd.set_value(df.loc[:, Xexpd_labels].as_matrix().astype(floatX))
    Xutil.set_value(df.loc[:, Xutil_labels].as_matrix().astype(floatX))
    stationid.set_value(df.loc[:, 'stationid'].as_matrix().astype(intX))

    df_count_choice = df[['stationid','choice', 'const']].groupby(['stationid','choice']).sum()
    idf_fe.set_value((1-np.isnan(df_count_choice.unstack().as_matrix())).astype(floatX))
    
    Xlelas.set_value(df.loc[:, Xlelas_labels].as_matrix().astype(floatX))
    Xlmu.set_value(df.loc[:, Xlmu_labels].as_matrix().astype(floatX))
    Xlsigma.set_value(df.loc[:, Xlsigma_labels].as_matrix().astype(floatX))


gammalelashat, gammalsigmahat, gammalmuhat, betaexpdhat, betautilhat, ltpconvehat, alphahat, betaexpdfehat, betautilfehat = getparams(thetahat, expdfe, utilfe)

def bootstrap(df, randskip, nbstr, prefix):
    np.random.seed(1234)
    for i in range(randskip):
        resampled_stationid = resample(np.arange(nstation),replace=True,n_samples=nstation)
        
    for i in range(randskip, randskip+nbstr):
        resampled_stationid = resample(np.arange(nstation),replace=True,n_samples=nstation)
        dfbstr = bootstrap_sample_by_station(df,resampled_stationid)        
        if utilfe:
            newbetautilfehat = betautilfehat[resampled_stationid]

        if expdfe:
            newbetaexpdfehat = np.concatenate([[[0]], betaexpdfehat], axis=0) + betaexpdhat[-1] #betaexpdhat[-1] is const
            newbetaexpdfehat = newbetaexpdfehat[resampled_stationid]
            betaexpdhat[-1] = newbetaexpdfehat[0]
            newbetaexpdfehat = newbetaexpdfehat - betaexpdhat[-1]

        newthetahat = np.hstack([a.ravel() for a in (gammalelashat, gammalsigmahat, gammalmuhat, betaexpdhat, betautilhat, 
                                                     ltpconvehat, alphahat, 
                                                     newbetaexpdfehat[1:] if expdfe else np.zeros([]), 
                                                     newbetautilfehat if utilfe else np.zeros([]))])
        
        set_shared_value(dfbstr)
        thetahat2 , _, _, _, nloglvalue, status = pyipopt.fmin_unconstrained(
            eval_f,
            newthetahat.astype(floatXnp),
            fprime=eval_g,
            fhess=eval_h,)

        res = [i, thetahat2, nloglvalue, status, eval_g(thetahat2)]
        with open(prefix + 'bstr{:0>4}.pkl'.format(i), 'wb') as f:
            pickle.dump(res, f)
        
        print('bootstrap sample {:0>4}, status {}'.format(i, status))

if len(sys.argv) > 2:
    skip = int(sys.argv[1])
    nbstr = int(sys.argv[2])    

    if not os.path.exists(bstr_prefix):
        os.makedirs(bstr_prefix)

    bootstrap(df,skip,nbstr,bstr_prefix)
