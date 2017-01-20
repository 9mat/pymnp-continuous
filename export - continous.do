use "D:\Dropbox\PhD Study\research\mnp_demand_gas_ethanol_dhl\individual_data_wide.dta", clear
set more off

drop if dv_rj
drop if choice == 4

keep consumerid choice value_total treattype /// 
		pg_km_adj pgmidgrade_km_adj pe_km_adj ///
		dv_ctb dv_bh dv_rec ///
		dv_female dv_age_25to40y dv_age_morethan65y dv_somesecondary dv_somecollege dv_carpriceadj_p75p100 dv_usageveh_p75p100 stationvisit_avgcarprice_adj
		
order consumerid choice value_total treattype /// 
		pg_km_adj pgmidgrade_km_adj pe_km_adj ///
		dv_ctb dv_bh dv_rec ///
		dv_female dv_age_25to40y dv_age_morethan65y dv_somesecondary dv_somecollege dv_carpriceadj_p75p100 dv_usageveh_p75p100 stationvisit_avgcarprice_adj		
outsheet using "D:\Dropbox\PhD Study\research\mnp_demand_gas_ethanol_dhl\continous\data_new_volume.csv", replace comma
