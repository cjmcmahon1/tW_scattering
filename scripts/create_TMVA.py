import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import awkward as ak
import pickle
from sklearn.preprocessing import quantile_transform, QuantileTransformer
import uproot
import glob
import os
#!source ~/scripts/renew_cms.sh
from yahist import Hist1D
import processor.BDT_analysis as BDT_analysis
#from coffea import processor, hist

import Tools.objects

import postProcessing.makeCards
import postProcessing.datacard_comparison.compare_datacards as compare_datacards
import uuid, os, uproot, shutil

import ROOT
print(xgb.__version__)
print(ROOT.TMVA.Experimental.SaveXGBoost)

files_all_categories = ["signal_tch.root", "signal_tuh.root", "fakes_mc.root", "flips_mc.root", "rares.root"]
dd_files = ["signal_tch.root", "signal_tuh.root", "data_fakes.root", "data_flips.root", "rares.root"]
HCT_cat = ["signal_tch.root", "fakes_mc.root", "flips_mc.root", "rares.root"]
HUT_cat = ["signal_tuh.root", "fakes_mc.root", "flips_mc.root", "rares.root"]
HCT_cat_dd = ["signal_tch.root", "data_fakes.root", "data_flips.root", "rares.root"]
HUT_cat_dd = ["signal_tuh.root", "data_fakes.root", "data_flips.root", "rares.root"]

all_files    = (["2016/MC/" + f for f in files_all_categories] + ["2017/MC/" + f for f in files_all_categories] + ["2018/MC/" + f for f in files_all_categories])
HCT_files    = (["2016/MC/" + f for f in HCT_cat] + ["2017/MC/" + f for f in HCT_cat] + ["2018/MC/" + f for f in HCT_cat])
HUT_files    = (["2016/MC/" + f for f in HUT_cat] + ["2017/MC/" + f for f in HUT_cat] + ["2018/MC/" + f for f in HUT_cat])

all_files_dd = (["2016/data_driven/" + f for f in dd_files] + ["2017/data_driven/" + f for f in dd_files] + ["2018/data_driven/" + f for f in dd_files])
HCT_files_dd = (["2016/data_driven/" + f for f in HCT_cat_dd] + ["2017/data_driven/" + f for f in HCT_cat_dd] + ["2018/data_driven/" + f for f in HCT_cat_dd])
HUT_files_dd = (["2016/data_driven/" + f for f in HUT_cat_dd] + ["2017/data_driven/" + f for f in HUT_cat_dd] + ["2018/data_driven/" + f for f in HUT_cat_dd])

input_baby_dir   = "/home/users/cmcmahon/fcnc/ana/analysis/helpers/BDT/babies/data_driven_v2/"
base_output_dir  = "/home/users/cmcmahon/public_html/BDT"

HCT              = BDT_analysis.BDT(input_baby_dir, HCT_files, base_output_dir, label="HCT_tmp", year="all")
HUT              = BDT_analysis.BDT(input_baby_dir, HUT_files, base_output_dir, label="HUT_tmp", year="all")

every_BDT = [HCT, HUT]

#for bdt in every_BDT:
HCT.gen_BDT_and_plot(load_BDT=True, optimize=False, retrain=True, plot=False)
ROOT.TMVA.Experimental.SaveXGBoost(HCT.classifier, "myBDT", "/home/users/cmcmahon/public_html/tmva101.root")