import pandas as pd

hg19_test1 = pd.read_csv("hg19_512_Y_1_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test2 = pd.read_csv("hg19_512_1_2_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test3 = pd.read_csv("hg19_512_2_3_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test4 = pd.read_csv("hg19_512_3_4_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test5 = pd.read_csv("hg19_512_4_5_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test6 = pd.read_csv("hg19_512_5_6_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test7 = pd.read_csv("hg19_512_6_7_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test8 = pd.read_csv("hg19_512_7_8_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test9 = pd.read_csv("hg19_512_8_9_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test10 = pd.read_csv("hg19_512_9_10_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test11 = pd.read_csv("hg19_512_10_11_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test12 = pd.read_csv("hg19_512_11_12_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test13 = pd.read_csv("hg19_512_12_13_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test14 = pd.read_csv("hg19_512_13_14_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test15 = pd.read_csv("hg19_512_14_15_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test16 = pd.read_csv("hg19_512_15_16_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test17 = pd.read_csv("hg19_512_16_17_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test18 = pd.read_csv("hg19_512_17_18_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test19 = pd.read_csv("hg19_512_18_19_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test20 = pd.read_csv("hg19_512_19_20_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test21 = pd.read_csv("hg19_512_20_21_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test22 = pd.read_csv("hg19_512_21_22_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_testX = pd.read_csv("hg19_512_22_X_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_testY = pd.read_csv("hg19_512_X_Y_performance_summary_with_validation2.csv").reset_index(names=["Algo"])
hg19_test = pd.concat([hg19_test1,hg19_test2,hg19_test3,hg19_test4,hg19_test5,hg19_test6,hg19_test7,hg19_test8,\
hg19_test9,hg19_test10,hg19_test11,hg19_test12,hg19_test13,hg19_test14,hg19_test15,hg19_test16,\
hg19_test17,hg19_test18,hg19_test19,hg19_test20,hg19_test21,hg19_test22,hg19_testX,hg19_testY],axis=0)
hg19_test.reset_index(drop=True,inplace=True)
rf_accs = []
rf_f1s =[]
rf_precs = []
rf_recs = []
xg_accs = []
xg_f1s =[]
xg_precs = []
xg_recs = []
lg_accs = []
lg_f1s =[]
lg_precs = []
lg_recs = []
chrs = []
print("hg19_test\n",hg19_test)

for ind in range(len(hg19_test.index)):
	test_acc = hg19_test["Test Accuracy"].loc[ind]
	test_f1 = hg19_test["Test F1-Score"].loc[ind]
	test_prec = hg19_test["Test Precision"].loc[ind]
	test_rec = hg19_test["Test Recall"].loc[ind]
	chr = "chr"+str(ind+1)
	chrs.append(chr)
	if hg19_test["Unnamed: 0"].loc[ind]=="Random Forest":
		rf_accs.append(test_acc)
		rf_f1s.append(test_f1)
		rf_precs.append(test_prec)
		rf_recs.append(test_rec)
	if hg19_test["Unnamed: 0"].loc[ind]=="XGBoost":
                xg_accs.append(test_acc)
                xg_f1s.append(test_f1)
                xg_precs.append(test_prec)
                xg_recs.append(test_rec)
	if hg19_test["Unnamed: 0"].loc[ind]=="LightGBM":
                lg_accs.append(test_acc)
                lg_f1s.append(test_f1)
                lg_precs.append(test_prec)
                lg_recs.append(test_rec)

rf_df = pd.DataFrame(data=list(zip(chrs,rf_accs,rf_f1s,rf_precs,rf_recs)),columns=["Chromosome","Accuracy","F1-Score","Precision","Recall"])
xg_df = pd.DataFrame(data=list(zip(chrs,xg_accs,xg_f1s,xg_precs,xg_recs)),columns=["Chromosome","Accuracy","F1-Score","Precision","Recall"])
lg_df = pd.DataFrame(data=list(zip(chrs,lg_accs,lg_f1s,lg_precs,lg_recs)),columns=["Chromosome","Accuracy","F1-Score","Precision","Recall"])

rf_df.to_csv("RF_LOCOCV.csv")
xg_df.to_csv("XG_LOCOCV.csv")
lg_df.to_csv("LG_LOCOCV.csv")

