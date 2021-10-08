import processor.BDT_analysis as BDT_analysis
import pandas as pd
import xgboost as xgb

BDT_features = ["Most_Forward_pt",
              "HT",
              "LeadLep_eta",
              "LeadLep_pt",
              "LeadLep_dxy",
              "LeadLep_dz",
              "SubLeadLep_pt",
              "SubLeadLep_eta",
              "SubLeadLep_dxy",
              "SubLeadLep_dz",
              "nJets",
              "nBtag",
              "LeadJet_pt",
              "SubLeadJet_pt",
              "SubSubLeadJet_pt",
              "LeadJet_BtagScore",
              "SubLeadJet_BtagScore",
              "SubSubLeadJet_BtagScore",
              "nElectron",
              "MET_pt",
              "LeadBtag_pt",
              "MT_LeadLep_MET",
              "MT_SubLeadLep_MET",
              "LeadLep_SubLeadLep_Mass",
              "SubSubLeadLep_pt",
              "SubSubLeadLep_eta",
              "SubSubLeadLep_dxy",
              "SubSubLeadLep_dz",
              "MT_SubSubLeadLep_MET",
              "LeadBtag_score",
              "Weight"]

def make_BDT_test_csv(booster_path, label="HCT"):
        booster = xgb.Booster() # init model
        booster.load_model(booster_path)  # load data
        test_df = pd.read_csv("test_events.csv")
        results = booster.predict(BDT_analysis.make_dmatrix(test_df, BDT_features))
        test_df["result"] = results
        test_df = test_df.drop(labels=["Weight", "Label", "Category"], axis=1)
        test_df.to_csv("python_test_results_{}.csv".format(label), index=False)
        print(results)
        
make_BDT_test_csv("booster_HCT.model")