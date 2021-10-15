import awkward as ak

from coffea import processor, hist
import numpy as np
import pandas as pd
from yahist import Hist1D, Hist2D
from Tools.helpers import mt
from Tools.fake_rate import fake_rate
from Tools.SS_selection import SS_selection
import production.weights

import uproot
import glob
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import quantile_transform, QuantileTransformer
import sklearn.utils
import uproot
import glob
import warnings
import os
import os.path
import time
from yahist import Hist1D

import Tools.objects
from sklearn.metrics import auc, roc_auc_score, roc_curve
import postProcessing.makeCards
import postProcessing.datacard_comparison.compare_datacards as compare_datacards

bins_dict = { "Most_Forward_pt":np.linspace(10,200,30),
              "HT":np.linspace(80,1200,30),
              "LeadLep_eta":np.linspace(0,2.4,15),
              "MET_pt":np.linspace(0,300,30),
              "LeadLep_pt":np.linspace(10,200,30),
              "LeadLep_dxy":np.linspace(0,0.015,30),
              "LeadLep_dz":np.linspace(0,0.015,30),
              "SubLeadLep_pt":np.linspace(10,120,30),
              "SubLeadLep_eta":np.linspace(0,2.4,15),
              "SubLeadLep_dxy":np.linspace(0,0.015,30),
              "SubLeadLep_dz":np.linspace(0,0.015,30),
              "nJets":np.linspace(0.5,8.5,9),
              "LeadJet_pt":np.linspace(25,450,30),
              "SubLeadJet_pt":np.linspace(10,250,30),
              "SubSubLeadJet_pt":np.linspace(10,250,30),
              "LeadJet_BtagScore":np.linspace(0, 1, 10),
              "SubLeadJet_BtagScore":np.linspace(0, 1, 10),
              "SubSubLeadJet_BtagScore":np.linspace(0, 1, 10),
              "nElectron":np.linspace(-0.5,4.5,6),
              "MT_LeadLep_MET":np.linspace(0,300,30),
              "MT_SubLeadLep_MET":np.linspace(0,300,30),
              "LeadBtag_pt":np.linspace(0,200,30),
              "nBtag":np.linspace(-0.5,3.5,5),
              "LeadLep_SubLeadLep_Mass":np.linspace(0, 350, 30),
              "SubSubLeadLep_pt":np.linspace(10,120,30),
              "SubSubLeadLep_eta":np.linspace(0,2.4,15),
              "SubSubLeadLep_dxy":np.linspace(0,0.015,30),
              "SubSubLeadLep_dz":np.linspace(0,0.015,30),
              "MT_SubSubLeadLep_MET":np.linspace(0,300,30),
              "LeadBtag_score":np.linspace(0, 1, 10),
              "LeadJet_CtagScore":np.linspace(0, 1, 10),
              "SubLeadJet_CtagScore":np.linspace(0, 1, 10),
              "SubSubLeadJet_CtagScore":np.linspace(0, 1, 10)
            }

tex_dict =  { "Most_Forward_pt":r'Most Forward $p_T$',
              "HT":r'$H_T$',
              "LeadLep_eta":r'Leading Lepton $\left|\eta\right|$',
              "MET_pt":r'MET $p_T$',
              "LeadLep_pt":r'Leading Lepton $p_T$',
              "LeadLep_dxy":r'Leading Lepton $d_{xy}$',
              "LeadLep_dz":r'Leading Lepton $d_{z}$',
              "SubLeadLep_pt":r'Sub-Leading Lepton $p_T$',
              "SubLeadLep_eta":r'Sub-Leading Lepton $\left|\eta\right|$',
              "SubLeadLep_dxy":"Sub-Leading Lepton $d_{xy}$",
              "SubLeadLep_dz":r'Sub-Leading Lepton $d{z}$',
              "nJets":r'Number of Jets',
              "LeadJet_pt":r'Leading Jet $p_T$',
              "SubLeadJet_pt":r'Sub-Leading Jet $p_T$',
              "SubSubLeadJet_pt":r'Sub-Sub-Leading Jet $p_T$',
              "LeadJet_BtagScore":r'Leading Jet Btag Score (deepFlavB)',
              "SubLeadJet_BtagScore":r'Sub-Leading Jet Btag Score(deepFlavB)',
              "SubSubLeadJet_BtagScore":r'Sub-Sub-Leading Jet Btag Score (deepFlavB)',
              "nElectron":r'Number of Electrons',
              "MT_LeadLep_MET":r'mt(Leading Lepton, MET)',
              "MT_SubLeadLep_MET":r'mt(Sub-Leading Lepton, MET)',
              "LeadBtag_pt":r'Leading B-Tagged Jet $p_T$',
              "nBtag":r'Number of B-Tagged Jets',
              "LeadLep_SubLeadLep_Mass":r'Leading Lepton + Sub-Leading Lepton Invariant Mass',
              "SubSubLeadLep_pt":r'Sub-Sub-Leading Lepton $p_T$',
              "SubSubLeadLep_eta":r'Sub-Sub-Leading Lepton $\left|\eta\right|$',
              "SubSubLeadLep_dxy":r'Sub-Sub-Leading Lepton $d_{xy}$',
              "SubSubLeadLep_dz":r'Sub-Sub-Leading Lepton $d_{z}$',
              "MT_SubSubLeadLep_MET":r'mt(Sub-Sub-Leading Lepton, MET)',
              "LeadBtag_score":r'Leading B-Tagged Jet Score (deepFlavB)',
              "LeadJet_CtagScore":r'Leading Jet Ctag Score (deepFlavC)',
              "SubLeadJet_CtagScore":r'Sub-Leading Jet Ctag Score(deepFlavC)',
              "SubSubLeadJet_CtagScore":r'Sub-Sub-Leading Jet Ctag Score (deepFlavC)'
            }

translate = { #for translating Jackson's pandas df into an equivalent BDT df
        "Most_Forward_pt":"forward_jet_pt",
        "HT":"HT",
        "LeadLep_eta":"lead_lep_eta",
        "MET_pt":"MET_pt",
        "LeadLep_pt":"lead_lep_pt",
        "LeadLep_dxy":"lead_lep_dxy",
        "LeadLep_dz":"lead_lep_dz",
        "SubLeadLep_pt":"sublead_lep_pt",
        "SubLeadLep_eta":"sublead_lep_eta",
        "SubLeadLep_dxy":"sublead_lep_dxy",
        "SubLeadLep_dz":"sublead_lep_dz",
        "nJets":"n_jets",
        "LeadJet_pt":"lead_jet_pt",
        "SubLeadJet_pt":"sublead_jet_pt",
        "SubSubLeadJet_pt":"subsublead_jet_pt",
        "LeadJet_BtagScore":"lead_jet_btag_score",
        "SubLeadJet_BtagScore":"sublead_jet_btag_score",
        "SubSubLeadJet_BtagScore":"subsublead_jet_btag_score",
        "nElectron":"n_electrons",
        "MT_LeadLep_MET":"lead_lep_MET_MT",
        "MT_SubLeadLep_MET":"sublead_lep_MET_MT",
        "LeadBtag_pt":"lead_btag_pt",
        "nBtag":"n_btags",
        "LeadLep_SubLeadLep_Mass":"sub_lead_lead_mass",
        "SubSubLeadLep_pt":"subsublead_lep_pt",
        "SubSubLeadLep_eta":"subsublead_lep_eta",
        "SubSubLeadLep_dxy":"subsublead_lep_dxy",
        "SubSubLeadLep_dz":"subsublead_lep_dz",
        "MT_SubSubLeadLep_MET":"subsublead_lep_MET_MT",
        "LeadBtag_score":"lead_btag_btag_score",
        "Weight":"weight"
}

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
#               "LeadJet_CtagScore",
#               "SubLeadJet_CtagScore",
#               "SubSubLeadJet_CtagScore",
              "Weight"]

train_features = BDT_features.copy()
train_features.pop(-1)

systematics_weights = [
    "Weight_LepSF_up",
    "Weight_LepSF_down",
    "Weight_Trigger_up",
    "Weight_Trigger_down",
    "Weight_PU_up",
    "Weight_PU_down",
    "Weight_bTag_up",
    "Weight_bTag_down"
]

def load_category(category, baby_dir="/home/users/cmcmahon/fcnc/ana/analysis/helpers/BDT/babies/2018/dilep/", year=None, BDT_features=BDT_features, systematics=False):
    file_name = baby_dir + "{}".format(category)
    if year != None:
        file_name += "_{}".format(year)
    file_name += ".root"
    try:
        tree = uproot.open(file_name)['T']
    except:
        tree = uproot.open(file_name)['Events']
    process_name = file_name[(file_name.rfind('/')+1):(file_name.rfind('.'))]
    tmp_df = pd.DataFrame()
    df_values = tree.arrays()
    for feature in BDT_features:
        tmp_df[feature] = np.array(df_values[feature])
    if systematics:
        for weight in systematics_weights:
            tmp_df[weight] = np.array(df_values[weight])
    if "signal" in process_name:
        tmp_df["Label"] = "s"
    else:
        tmp_df["Label"] = "b"
    tmp_df["Category"] = category
    return tmp_df

def BDT_train_test_split(full_data, verbose=True):
    if verbose:
        print('Size of data: {}'.format(full_data.shape))
        print('Number of events: {}'.format(full_data.shape[0]))
        print('Number of columns: {}'.format(full_data.shape[1]))

        print ('\nList of features in dataset:')
        for col in full_data.columns:
            print(col)
        # look at column labels --- notice last one is "Label" and first is "EventId" also "Weight"
        print('Number of signal events: {}'.format(len(full_data[full_data.Label == 's'])))
        print('Number of background events: {}'.format(len(full_data[full_data.Label == 'b'])))
        print('Fraction signal: {}'.format(len(full_data[full_data.Label == 's'])/(float)(len(full_data[full_data.Label == 's']) + len(full_data[full_data.Label == 'b']))))

    full_data['Label'] = full_data.Label.astype('category')
    (data_train, data_test) = train_test_split(full_data, train_size=0.5)
    #training table
    print("Training Event Counts:")
    print("Signal: {}\nFakes: {}\nFlips: {}\nRares: {}".format(len(data_train[data_train.Label=="s"]),len(data_train[data_train.Category=="fakes_mc"]),
                                                              len(data_train[data_train.Category=="flips_mc"]), len(data_train[data_train.Category=="rares"])))
    print("Test Event Counts:")
    print("Signal: {}\nFakes: {}\nFlips: {}\nRares: {}".format(len(data_test[data_test.Label=="s"]),len(data_test[data_test.Category=="fakes_mc"]),
                                                              len(data_test[data_test.Category=="flips_mc"]), len(data_test[data_test.Category=="rares"])))
    if verbose:
        print('Number of training samples: {}'.format(len(data_train)))
        print('Number of testing samples: {}'.format(len(data_test)))

        print('\nNumber of signal events in training set: {}'.format(len(data_train[data_train.Label == 's'])))
        print('Number of background events in training set: {}'.format(len(data_train[data_train.Label == 'b'])))
        print('Fraction signal: {}'.format(len(data_train[data_train.Label == 's'])/(float)(len(data_train[data_train.Label == 's']) + len(data_train[data_train.Label == 'b']))))
        
    return data_train, data_test

def gen_BDT(signal_name, param, num_trees, output_dir, bdt_features, booster_name="", data_train=None, data_test=None, flag_load=False, verbose=True, flag_save_booster=True):
    if booster_name == "":
        booster_path = output_dir + "booster_{}.model".format(signal_name)
    else:
        booster_path = output_dir + "booster_{}.model".format(booster_name)
        
    if flag_load:
        print("Loading saved model...")
        booster = xgb.Booster({"nthread": 4})  # init model xgb.Booster()
        booster.load_model(booster_path)  # load data
        data_train = pd.read_csv(output_dir + "data_train.csv")
        data_test = pd.read_csv(output_dir + "data_test.csv")
        data_train['Label'] = data_train.Label.astype('category')
        data_test['Label'] = data_test.Label.astype('category')
        
    assert (type(data_train) != type(None) and type(data_test) != type(None))
    feature_names = bdt_features.copy()
    feature_names.pop(-1)
    train_weights = data_train.Weight
    test_weights = data_test.Weight
    # we skip the first and last two columns because they are the ID, weight, and label
    train = xgb.DMatrix(data=data_train[feature_names],label=data_train.Label.cat.codes,
                        missing=-999.0,feature_names=feature_names, weight=np.abs(train_weights))
    test = xgb.DMatrix(data=data_test[feature_names],label=data_test.Label.cat.codes,
                       missing=-999.0,feature_names=feature_names, weight=np.abs(test_weights))
    evals_result = {}

    if verbose:
        print(feature_names)
        print(data_test.Label.cat.codes)
        print("weights:\n")
        print(train_weights)

    else:
        if verbose:
            print("Training new model...")
        evals = [(train, "train"), (test,"test")]
        booster = xgb.train(param,train,num_boost_round=num_trees, evals=evals, evals_result=evals_result, verbose_eval=False, early_stopping_rounds=min(3,(num_trees//10)))
        if verbose:
            print(booster.eval(test))

    #if the tree is of interest, we can save it
    if flag_save_booster:
        booster.save_model(booster_path)
        data_train.to_csv(output_dir + "data_train.csv")
        data_test.to_csv(output_dir + "data_test.csv")

    return booster, train, test, evals_result

def optimize_BDT_params(data_train, bdt_features, n_iter=20, num_folds=3, param_grid={}):
    y_train = data_train.Label.cat.codes
    feature_names = bdt_features.copy()
    feature_names.pop(-1)
    x_train = data_train[feature_names]atrix
    clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic', eval_metric="logloss", use_label_encoder=False)
    if len(param_grid.keys())==0:
        param_grid = {
                      'max_depth': [3, 4, 5, 6],
                      'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7],
                      'subsample': [0.7, 0.8, 0.9, 1.0],
                      'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                      'colsample_bylevel': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                      'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
                      'gamma': [0, 0.25, 0.5, 1.0],
                      'reg_lambda': [0.1, 1.0, 5.0, 10.0],
                      'n_estimators': [20, 40, 50, 60, 100, 150, 200]
                     }
    fit_params = {'eval_metric': 'mlogloss',
                  'early_stopping_rounds': 10}
    rs_clf = RandomizedSearchCV(clf_xgb, param_grid, n_iter=n_iter,
                                n_jobs=1, verbose=2, cv=num_folds,
                                refit=False)
    print("Randomized search..")
    search_time_start = time.time()
    rs_clf.fit(x_train, y_train)
    print("Randomized search time:", time.time() - search_time_start)

    best_score = rs_clf.best_score_
    best_params = rs_clf.best_params_
    return best_params

#plotting functions
def make_yahist(x):
    x_counts = x[0]
    x_bins = x[1]
    yahist_x = Hist1D.from_bincounts(x_counts, x_bins)
    return yahist_x

def make_dmatrix(df, feature_names=None):
    #breakpoint()
    if feature_names==None:
        feature_names = train_features#df.columns[:-2]
    else:
        feature_names = feature_names.copy()
        feature_names.pop(-1)
    df["Label"] = df.Label.astype("category")
    return xgb.DMatrix(data=df[feature_names],label=df.Label.cat.codes, missing=-999.0)#feature_names=feature_names)

def get_s_b_ratio(s_hist, b_hist):
    tot_hist = s_hist + b_hist
    denom_hist = Hist1D.from_bincounts(np.sqrt(np.array(tot_hist.counts)), bins=tot_hist.edges)
    result = s_hist.divide(denom_hist)
    return result

def gen_hist(data_test, label, out_dir, savefig=False, plot=False):
    plt.figure(label, figsize=(6,6));
    bins = bins_dict[label]
    values_signal = data_test[label][data_test.Label == 's']
    weights_signal = data_test["Weight"][data_test.Label == 's']
    values_background = data_test[label][data_test.Label == 'b']
    weights_background = data_test["Weight"][data_test.Label == 'b']
    weight_ratio = np.sum(weights_signal) / np.sum(weights_background)
    weights_signal = weights_signal / weight_ratio
    if label not in ["nJet", "nBtag", "nElectron", "SubSubLeadLep_pt", "SubSubLeadLep_eta", "SubSubLeadLep_dz", 
                     "SubSubLeadLep_dxy", "SubLeadJet_pt", "SubSubLeadJet_pt", "SubLeadJet_BtagScore", 
                     "SubSubLeadJet_BtagScore", "LeadBtag_pt", "LeadBtag_score", "MT_SubSubLeadLep_MET"]: #features where we want overflow/underflow bins
        plt.hist(np.clip(values_signal, bins[0], bins[-1]),bins=bins_dict[label],
                 histtype='step',color='midnightblue',label='signal', weights=np.clip(weights_signal, bins[0], bins[-1]), density=True)#np.clip(weights_signal, bins[0], bins[-1])
        plt.hist(np.clip(values_background, bins[0], bins[-1]),bins=bins_dict[label],
                 histtype='step',color='firebrick',label='background', weights=np.clip(weights_background, bins[0], bins[-1]), density=True)#np.clip(weights_background, bins[0], bins[-1])
    else: #do not make an overflow/underflow bin
        plt.hist(values_signal, bins=bins_dict[label],
                 histtype='step',color='midnightblue',label='signal', weights=weights_signal)#density=True);
        plt.hist(values_background,bins=bins_dict[label],
                 histtype='step',color='firebrick',label='background', weights=weights_background)#density=True);

    log_params = ["SubLeadLep_dxy", "SubLeadLep_dz", "LeadLep_dxy", "LeadLep_dz", "SubSubLeadLep_dz", "SubSubLeadLep_dxy"]
    if label in log_params:
        plt.yscale("log")
    plt.xlabel(tex_dict[label]);
    plt.ylabel('Yield',fontsize=12);
    plt.legend(frameon=False);
    if savefig:
        plt.savefig(out_dir + "histograms/{}.pdf".format(label))
        plt.savefig(out_dir + "histograms/{}.png".format(label))
    if plot:
        plt.draw()
    else:
        plt.close()
        
def get_SR_BR(signal_name, base_dir, version, year, BDT_features, background_category="none", flag_match_yields=True, gen_signal=True, gen_background=True):
    desired_output.update({"BDT_df": processor.column_accumulator(np.zeros(shape=(0,len(BDT_features)+1)))})
    if signal_name == "HCT":
        signal_files = glob.glob(base_dir + "*hct*.root")
    elif signal_name == "HUT":
        signal_files = glob.glob(base_dir + "*hut*.root")
    elif signal_name == "combined_HCT_HUT":
        signal_files = glob.glob(base_dir + "*hct*.root") + glob.glob(base_dir + "*hut*.root")
        
    if background_category=="none":
        remove_files = glob.glob(base_dir + "*hut*.root") + glob.glob(base_dir + "*hct*.root") + glob.glob(base_dir + "*data.root") + glob.glob(base_dir + "tt[0-9]lep.root")
        background_files = glob.glob(base_dir + "*.root")
        [background_files.remove(r) for r in remove_files]
    elif background_category=="fakes":
        background_files = [base_dir + "ttjets.root", base_dir + "wjets.root", base_dir + "ttg_1lep.root"]
    elif background_category=="flips":
        background_files = [base_dir + f for f in ["dyjets_m10-50.root", "dyjets_m50.root", "zg.root", "tw_dilep.root"]]#, "ww.root"]]
    elif background_category=="rares":
        remove_files = [base_dir + f for f in ["dyjets_m10-50.root", "dyjets_m50.root", "ww.root", "zg.root", "ttjets.root", "wjets.root", "ttg_1lep.root", "tw_dilep.root"]]
        remove_files += glob.glob(base_dir + "*hut*.root") + glob.glob(base_dir + "*hct*.root") + glob.glob(base_dir + "*data.root") + glob.glob(base_dir + "tt[0-9]lep.root")
        background_files = glob.glob(base_dir + "*.root")
        [background_files.remove(r) for r in remove_files]
    #print(background_files)
    concat_file = []
    for file in background_files:
        concat_file.append(file[file.rfind('/')+1:file.rfind('.')])
    print(concat_file)
    signal_fileset = {}
    for s in signal_files:
        process_name = s[(s.rfind('/')+1):(s.rfind('.'))]
        signal_fileset[process_name] = [s]
    background_fileset = {}    
    for b in background_files:
        process_name = b[(b.rfind('/')+1):(b.rfind('.'))]
        background_fileset[process_name] = [b]

    exe_args = {
        'workers': 16,
        'function_args': {'flatten': False},
        "schema": NanoAODSchema,
        "skipbadfiles": True,
    }
    exe = processor.futures_executor
    with warnings.catch_warnings(): #Ignoring all RuntimeWarnings (there are a lot)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if gen_signal:
            signal_output = processor.run_uproot_job(
                signal_fileset,
                "Events",
                nano_analysis(year=year, variations=[], accumulator=desired_output, BDT_features=BDT_features, version=version),
                exe,
                exe_args,
                chunksize=250000,
            )
            signal_BDT_params = pd.DataFrame(data=signal_output["BDT_df"].value, columns=(["event"]+BDT_features))
            signal_BDT_params["Label"] = "s"
        else:
            signal_BDT_params = None
            
        if gen_background:
            background_output = processor.run_uproot_job(
                background_fileset,
                "Events",
                nano_analysis(year=year, variations=[], accumulator=desired_output, BDT_features=BDT_features, version=version),
                exe,
                exe_args,
                chunksize=250000,
            )
            background_BDT_params = pd.DataFrame(data=background_output["BDT_df"].value, columns=(["event"]+BDT_features))
            background_BDT_params["Label"] = "b"
            background_BDT_params = sklearn.utils.shuffle(background_BDT_params) #shuffle our background before we cut out a subset of it
    #background_BDT_params = background_BDT_params[:signal_BDT_params.shape[0]] #make the background only as large as the signal
        else:
            background_BDT_params = None
    if flag_match_yields and gen_signal and gen_background:
        signal_yield = sum(signal_BDT_params.Weight)
        background_yield = sum(background_BDT_params.Weight)
        SR_BR_ratio = signal_yield/background_yield
        print("signal yield:{}".format(signal_yield))
        print("background yield:{}".format(background_yield))
        print("SR/BR yield ratio:{}".format(SR_BR_ratio))
        signal_BDT_params.Weight = signal_BDT_params.Weight / SR_BR_ratio
        print("new signal yield:{}".format(sum(signal_BDT_params.Weight)))
        print("new SR/BR yield ratio:{}".format(sum(signal_BDT_params.Weight) / background_yield))
    if gen_signal and gen_background:
        full_data = pd.concat([signal_BDT_params, background_BDT_params], axis=0)
    else:
        full_data = None
    return (signal_BDT_params, background_BDT_params, full_data)

def process_file(fname, base_dir, BDT_features, version, year):
    fileset = {}
    process_name = fname[(fname.rfind('/')+1):(fname.rfind('.'))]
    fileset[process_name] = [fname]
    desired_output.update({"BDT_df": processor.column_accumulator(np.zeros(shape=(0,len(BDT_features)+1)))})
    exe_args = {
        'workers': 16,
        'function_args': {'flatten': False},
        "schema": NanoAODSchema,
        "skipbadfiles": True,
    }
    exe = processor.futures_executor
    with warnings.catch_warnings(): #Ignoring all RuntimeWarnings (there are a lot)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tmp_output = processor.run_uproot_job(
            fileset,
            "Events",
            nano_analysis(year=year, variations=[], accumulator=desired_output, BDT_features=BDT_features, version=version),
            exe,
            exe_args,
            chunksize=250000,
        )
    tmp_BDT_params = pd.DataFrame(data=tmp_output["BDT_df"].value, columns=(["event"]+BDT_features))
    return tmp_BDT_params


class BDT:
    def __init__(self, in_base_dir, in_files=[], out_base_dir="/home/users/cmcmahon/public_html/BDT", label="tmp", year="all", booster=None, booster_label="", booster_params=None, train_predictions=None, test_predictions=None, BDT_features=BDT_features, pd_baby=False, pd_sig="HCT", train_files=None, test_files=None):
        self.in_base_dir = in_base_dir #"/home/users/cmcmahon/fcnc/ana/analysis/helpers/BDT/babies/2018"
        self.out_base_dir = out_base_dir #/home/users/cmcmahon/public_html/BDT
        self.train_files = train_files
        self.test_files = test_files
        self.in_files = in_files
        self.label = label
        self.year = year
        self.booster = booster
        self.booster_label = booster_label
        self.booster_params = booster_params
        self.train_predictions = train_predictions
        self.test_predictions = test_predictions
        self.BDT_features = BDT_features
        self.pd_baby = pd_baby
        self.pd_sig = pd_sig
        self.num_trees = 10
        if (type(train_files) != type(None) and (type(test_files) != type(None))): #if we specify a directory for training/testing datasets, don't split the data further
            train_baby_dir = [self.in_base_dir + f for f in train_files]
            train_sig, train_back, train_full = self.load_SR_BR_from_babies(train_baby_dir)
            test_baby_dir = [self.in_base_dir + f for f in test_files]
            test_sig, test_back, test_full = self.load_SR_BR_from_babies(test_baby_dir)
            train_full['Label'] = train_full.Label.astype('category')
            test_full['Label'] = test_full.Label.astype('category')
            #breakpoint()
            self.signal = pd.concat([train_sig, test_sig], axis=0)
            self.background = pd.concat([train_back, test_back], axis=0)
            self.full_data = pd.concat([self.signal, self.background], axis=0)
            self.signal['Label'] = self.signal.Label.astype('category')
            self.background['Label'] = self.background.Label.astype('category')
            self.full_data['Label'] = self.full_data.Label.astype('category')
            eq = self.equalize_yields(train_full, test_full, verbose=True)
            self.train_data = eq[0]
            self.test_data  = eq[1]
        else:
            self.signal, self.background, self.full_data = self.__get_signal_background__(pd_baby, in_base_dir, pd_sig)
            self.train_data, self.test_data = self.__train_test_split__()
        self.evals_result = None
        self.train_predictions = None
        self.test_predictions = None
        self.category_dict = None
        self.HCT_dict = None
        self.HUT_dict = None
        
    def combine_BDTs(self, other_BDTs, new_label, year="combined"):
        other = BDT(self.in_base_dir, self.in_files, self.out_base_dir, new_label, year=year)
        other.category_dict = self.category_dict.copy()
        other.HCT_dict = self.HCT_dict.copy()
        other.HUT_dict = self.HUT_dict.copy()
        for o in other_BDTs:
            for cat_key in ["signal", "fakes", "flips", "rares"]:
                other.category_dict[cat_key]["data"] = pd.concat([other.category_dict[cat_key]["data"], o.category_dict[cat_key]["data"]])
                other.HCT_dict[cat_key]["data"] = pd.concat([other.HCT_dict[cat_key]["data"], o.HCT_dict[cat_key]["data"]])
                other.HUT_dict[cat_key]["data"] = pd.concat([other.HUT_dict[cat_key]["data"], o.HUT_dict[cat_key]["data"]])
                other.category_dict[cat_key]["prediction"] = np.concatenate([other.category_dict[cat_key]["prediction"], o.category_dict[cat_key]["prediction"]])
                other.HCT_dict[cat_key]["prediction"] = np.concatenate([other.HCT_dict[cat_key]["prediction"], o.HCT_dict[cat_key]["prediction"]])
                other.HUT_dict[cat_key]["prediction"] = np.concatenate([other.HUT_dict[cat_key]["prediction"], o.HUT_dict[cat_key]["prediction"]])
        return other
        
    def load_pd_baby(self, pd_file_path, signal="HCT"): #for reading Jackson's babies
        baby = pd.read_hdf(pd_file_path)
        signal_BDT_params = pd.DataFrame()
        background_BDT_params = pd.DataFrame()
        if signal == "HCT":
            sig_label = -2
        elif signal == "HUT":
            sig_label = -1
            
        sig_df = baby[baby.label==sig_label]
        fakes_df = baby[baby.label>=3]
        flips_df = baby[baby.label==2]
        rares_df = baby[baby.label==1]
        back_df = pd.concat([fakes_df, flips_df, rares_df], axis=0)
        for feature in self.BDT_features:
            signal_BDT_params[feature] = baby[baby.label==sig_label][translate[feature]]
            background_BDT_params[feature] = baby[baby.label>=0][translate[feature]]
        signal_BDT_params["Label"] = "s"
        signal_BDT_params["Weight"] *= 100
        background_BDT_params["Label"] = "b"
        full_data = pd.concat([signal_BDT_params, background_BDT_params], axis=0)
        return signal_BDT_params, background_BDT_params, full_data
            
    def load_SR_BR_from_babies(self, baby_files): 
        #breakpoint()
        signal_BDT_params = pd.DataFrame()
        background_BDT_params = pd.DataFrame()
        for file in baby_files:
            tree = uproot.open(file)['T']
            process_name = file[(file.rfind('/')+1):(file.rfind('.'))]
            df = pd.DataFrame()
            df_values = tree.arrays()
            for feature in self.BDT_features:
                df[feature] = np.array(df_values[feature])
                
            if "signal" in process_name:
                df["Category"] = "signal"
                signal_BDT_params = pd.concat([signal_BDT_params, df], axis=0)
            else:
                df["Category"] = process_name
                background_BDT_params = pd.concat([background_BDT_params, df], axis=0)
            
        signal_BDT_params["Label"] = "s"
        background_BDT_params["Label"] = "b"
        full_data = pd.concat([signal_BDT_params, background_BDT_params], axis=0)
        return signal_BDT_params, background_BDT_params, full_data
    
    def __get_signal_background__(self, pd_baby=False, pd_dir = "", sig="HCT"):
        if pd_baby:
            return self.load_pd_baby(pd_dir, signal=sig)
        else:
            baby_dir = [self.in_base_dir + f for f in self.in_files]
            return self.load_SR_BR_from_babies(baby_dir)
    
    def equalize_yields(self, sig, back, verbose=False): #generate a copy of the signal/background babies where the signal yield is weighted down to be == background yield.
        sig_copy = sig.copy()
        back_copy = back.copy()
        signal_yield = sum(sig_copy.Weight)
        background_yield = sum(back_copy.Weight)
        SR_BR_ratio = signal_yield/background_yield
        if verbose:
            print("signal yield:{}".format(signal_yield))
            print("background yield:{}".format(background_yield))
            print("SR/BR yield ratio:{}".format(SR_BR_ratio))
        sig_copy.Weight = sig_copy.Weight / SR_BR_ratio
        if verbose:
            print("new signal yield:{}".format(sum(sig_copy.Weight)))
            print("new SR/BR yield ratio:{}".format(sum(sig_copy.Weight) / background_yield))
        full_data = pd.concat([sig_copy, back_copy], axis=0)
        return sig_copy, back_copy, full_data
    
    def __train_test_split__(self):
        data_train, data_test = BDT_train_test_split(self.equalize_yields(self.signal, self.background)[2], verbose=False)
        return data_train, data_test

    def optimize_booster(self, label=None):
        if label==None:
            self.booster_label = "random_search_optimized_{}".format(self.year)
        else:
            self.booster_label = label
        output_dir = "{0}/{1}/{2}/".format(self.out_base_dir, self.label, self.booster_label)
        self.booster_params = optimize_BDT_params(self.train_data, self.BDT_features)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir + "histograms", exist_ok=True)
        pickle.dump(self.booster_params, open(output_dir+"params.p", "wb"))
        
    def set_booster_label(self, label=None):
        if label==None:
            self.booster_label = "random_search_optimized_{}".format(self.year)
        
    def load_booster_params(self, label=None):
        if label==None:
            b_label = "random_search_optimized_{}".format(self.year)
        output_dir = "{0}/{1}/{2}/".format(self.out_base_dir, self.label, b_label)
        self.booster_params = pickle.load(open(output_dir + "params.p", "rb"))
    
    def get_predictions(self):
        self.test_predictions = self.booster.predict(self.test_Dmatrix)
        self.train_predictions = self.booster.predict(self.train_Dmatrix)
        
    def gen_BDT(self, flag_load=False, verbose=False, flag_save_booster=True):
        #breakpoint()
        output_dir = "{0}/{1}/{2}/".format(self.out_base_dir, self.label, self.booster_label)
        param = self.booster_params.copy()
        if verbose:
            print(param)
        self.num_trees = param.pop("n_estimators")
        param['objective']   = 'binary:logistic' # objective function
        param['eval_metric'] = 'error'           # evaluation metric for cross validation
        param = list(param.items()) + [('eval_metric', 'logloss')] + [('eval_metric', 'rmse')]
        if verbose:
            print(output_dir)
        booster, train, test, evals_result = gen_BDT(self.label, param, self.num_trees, output_dir, self.BDT_features, booster_name=self.label, data_train=self.train_data,
                                                     data_test=self.test_data, flag_load=flag_load, verbose=False, flag_save_booster=flag_save_booster)
        self.booster = booster
        self.train_Dmatrix = train
        self.test_Dmatrix = test
        self.evals_result = evals_result
        return self.booster
    
    def plot_ratio(self, region="signal", savefig=False, plot=True, verbose=False):
        bins = np.linspace(0, 1, 25)
        output_dir = "{0}/{1}/{2}/".format(self.out_base_dir, self.label, self.booster_label)
        if region == "signal":
            train_set_predictions = self.train_predictions[self.train_Dmatrix.get_label().astype(bool)]
            test_set_predictions  = self.test_predictions[self.test_Dmatrix.get_label().astype(bool)]
            train_weights = self.train_Dmatrix.get_weight()[self.train_Dmatrix.get_label().astype(bool)]
            test_weights = self.test_Dmatrix.get_weight()[self.test_Dmatrix.get_label().astype(bool)]
            title = "Signal Region Prediction from BDT"
            fig_directory = output_dir + "Prediction_Signal"

        elif region == "background":
            train_set_predictions = self.train_predictions[~(self.train_Dmatrix.get_label().astype(bool))]
            test_set_predictions  = self.test_predictions[~(self.test_Dmatrix.get_label().astype(bool))]
            train_weights = self.train_Dmatrix.get_weight()[~(self.train_Dmatrix.get_label().astype(bool))]
            test_weights = self.test_Dmatrix.get_weight()[~(self.test_Dmatrix.get_label().astype(bool))]
            title = "Background Region Prediction from BDT"
            fig_directory = output_dir + "Prediction_Background"

        #NEED TO FIX ERRORS WHEN NORMALIZING (to accomodate different sized train/test sets)
        if region == "signal" or region == "background":
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,8), gridspec_kw=dict(height_ratios=[3, 1]))
            hist_train = ax[0].hist(train_set_predictions, bins=bins, histtype='step',color='lime', label='training {}'.format(region), weights=train_weights)#, density=True)
            hist_test  = ax[0].hist(test_set_predictions,  bins=bins, histtype='step',color='magenta', label='test {}'.format(region), weights=test_weights)#, density=True)
            yahist_train = make_yahist(hist_train)
            yahist_test = make_yahist(hist_test)
            ratio = yahist_train.divide(yahist_test)
            ratio.plot(ax=ax[1], errors=True)
            ax[1].plot([0, 1], [1, 1], 'r')
            ax[1].set_ylim([0, 2])
            ax[0].set_xlabel(title,fontsize=12)
            ax[0].set_ylabel('Events',fontsize=12)
            ax[0].legend(frameon=False)
            if verbose:
                print("train area = {}, test area = {}".format(sum(hist_train[0] * np.diff(hist_train[1])), sum(hist_test[0] * np.diff(hist_test[1]))))
            if savefig:
                plt.savefig(fig_directory + ".pdf")
                plt.savefig(fig_directory + ".png")
            if plot:
                plt.draw()
            else:
                plt.close()
        elif region == "both":
            fig_directory = output_dir + "Prediction_All"
            
            sig_train_predictions = self.train_predictions[self.train_Dmatrix.get_label().astype(bool)]
            sig_test_predictions  = self.test_predictions[self.test_Dmatrix.get_label().astype(bool)]
            sig_train_weights = self.train_Dmatrix.get_weight()[self.train_Dmatrix.get_label().astype(bool)]
            sig_test_weights = self.test_Dmatrix.get_weight()[self.test_Dmatrix.get_label().astype(bool)]
            
            back_train_predictions = self.train_predictions[~(self.train_Dmatrix.get_label().astype(bool))]
            back_test_predictions  = self.test_predictions[~(self.test_Dmatrix.get_label().astype(bool))]
            back_train_weights = self.train_Dmatrix.get_weight()[~(self.train_Dmatrix.get_label().astype(bool))]
            back_test_weights = self.test_Dmatrix.get_weight()[~(self.test_Dmatrix.get_label().astype(bool))]
            plt.figure("sig/back", figsize=(7,7))
            hist_sig_train = Hist1D(sig_train_predictions, bins=bins, weights=sig_train_weights, color="midnightblue", label="signal (train)")
            hist_sig_test = Hist1D(sig_test_predictions, bins=bins, weights=sig_test_weights, color="blue", label="signal (test)")
            hist_back_train = Hist1D(back_train_predictions, bins=bins, weights=back_train_weights, color="firebrick", label="background (train)")
            hist_back_test = Hist1D(back_test_predictions, bins=bins, weights=back_test_weights, color="red", label="background (test)")
            hist_sig_train.plot(ax=plt.gca(), errors=True)
            hist_sig_test.plot(ax=plt.gca(), errors=True)
            hist_back_train.plot(ax=plt.gca(), errors=True)
            hist_back_test.plot(ax=plt.gca(), errors=True)
#             hist_sig_train = plt.hist(sig_train_predictions, bins=bins, histtype='step',color='midnightblue', label='signal (train)', weights=sig_train_weights)
#             hist_sig_test  = plt.hist(sig_test_predictions,  bins=bins, histtype='step',color='midnightblue', hatch="/", label='signal (test)', weights=sig_test_weights)
#             hist_back_train = plt.hist(back_train_predictions, bins=bins, histtype='step',color='firebrick', label='background (train)', weights=back_train_weights)
#             hist_back_test  = plt.hist(back_test_predictions,  bins=bins, histtype='step',color='firebrick', hatch="/", label='background (test)', weights=back_test_weights)
            plt.title("{} BDT Prediction of Signal and Background".format(self.label))
            plt.xlabel("BDT Score")
            plt.ylabel("Yield")
            plt.legend()
            if savefig:
                plt.savefig(fig_directory + ".pdf")
                plt.savefig(fig_directory + ".png")
            if plot:
                plt.draw()
            else:
                plt.close()
            
    def gen_prediction_plots(self, savefig=True, plot=False):
        test_weights = self.test_Dmatrix.get_weight()
        train_weights = self.train_Dmatrix.get_weight()
        out_dir = "{0}/{1}/{2}/".format(self.out_base_dir, self.label, self.booster_label)
        # plot all predictions (both signal and background)
        plt.figure();
        plt.hist(self.test_predictions,bins=np.linspace(0,1,30),histtype='step',color='darkgreen',label='All events', weights=test_weights);
        # make the plot readable
        plt.xlabel('Test Set Prediction from BDT',fontsize=12);
        plt.ylabel('Weighted Events',fontsize=12);
        plt.legend(frameon=False);
        if savefig:
            plt.savefig(out_dir + "Prediction_Total.pdf")
            plt.savefig(out_dir + "Prediction_Total.png")
        if plot:
            plt.draw()
        else:
            plt.close()
        # plot signal and background separately
        plt.figure();
        plt.hist(self.test_predictions[self.test_Dmatrix.get_label().astype(bool)],bins=np.linspace(0,1,30),
                 histtype='step',color='midnightblue',label='signal', density=True, weights=test_weights[self.test_Dmatrix.get_label().astype(bool)]);
        plt.hist(self.test_predictions[~(self.test_Dmatrix.get_label().astype(bool))],bins=np.linspace(0,1,30),
                 histtype='step',color='firebrick',label='background', density=True, weights=test_weights[~(self.test_Dmatrix.get_label().astype(bool))]);
        # make the plot readable
        plt.xlabel('Test Set Prediction from BDT',fontsize=12);
        plt.ylabel('Normalized Events',fontsize=12);
        plt.legend(frameon=False);
        if savefig:
            plt.savefig(out_dir + "Prediction_SR_BR.pdf")
            plt.savefig(out_dir + "Prediction_SR_BR.png")
        if plot:
            plt.draw()
        else:
            plt.close()

        plt.figure();
        plt.hist(self.train_predictions[self.train_Dmatrix.get_label().astype(bool)],bins=np.linspace(0,1,30),
                 histtype='step',color='midnightblue',label='signal', density=True, weights=train_weights[self.train_Dmatrix.get_label().astype(bool)]);
        plt.hist(self.train_predictions[~(self.train_Dmatrix.get_label().astype(bool))],bins=np.linspace(0,1,30),
                 histtype='step',color='firebrick',label='background', density=True, weights=train_weights[~(self.train_Dmatrix.get_label().astype(bool))]);
        # make the plot readable
        plt.xlabel('Training Set Prediction from BDT',fontsize=12);
        plt.ylabel('Normalized Events',fontsize=12);
        plt.legend(frameon=False);
        if savefig:
            plt.savefig(out_dir + "Training_Prediction_SR_BR.pdf")
            plt.savefig(out_dir + "Training_Prediction_SR_BR.png")
        if plot:
            plt.draw()
        else:
            plt.close()

        self.plot_ratio("signal", savefig, plot, verbose=False)
        self.plot_ratio("background", savefig, plot, verbose=False)
        self.plot_ratio("both", savefig, plot, verbose=False)
        cuts = np.linspace(0,1,200);
        train_TP = np.zeros(len(cuts))
        train_FP = np.zeros(len(cuts))
        train_TN = np.zeros(len(cuts))
        train_FN = np.zeros(len(cuts))
        test_TP = np.zeros(len(cuts))
        test_FP = np.zeros(len(cuts))
        test_TN = np.zeros(len(cuts))
        test_FN = np.zeros(len(cuts))
        # TPR = TP / (TP + FN)
        # FPR = FP / (FP + TN)
        for i,cut in enumerate(cuts):
            train_pos = (self.train_predictions >  cut)
            train_neg = (self.train_predictions <= cut)
            train_TP[i] = np.sum(self.train_Dmatrix.get_weight()[(self.train_data.Label=='s') & train_pos])
            train_FP[i] = np.sum(self.train_Dmatrix.get_weight()[(self.train_data.Label=='b') & train_pos])
            train_TN[i] = np.sum(self.train_Dmatrix.get_weight()[(self.train_data.Label=='b') & train_neg])
            train_FN[i] = np.sum(self.train_Dmatrix.get_weight()[(self.train_data.Label=='s') & train_neg])

            test_pos = (self.test_predictions >  cut)
            test_neg = (self.test_predictions <= cut)
            test_TP[i] = np.sum(self.test_Dmatrix.get_weight()[(self.test_data.Label=='s') & test_pos])
            test_FP[i] = np.sum(self.test_Dmatrix.get_weight()[(self.test_data.Label=='b') & test_pos])
            test_TN[i] = np.sum(self.test_Dmatrix.get_weight()[(self.test_data.Label=='b') & test_neg])
            test_FN[i] = np.sum(self.test_Dmatrix.get_weight()[(self.test_data.Label=='s') & test_neg])

        # plot efficiency vs. purity (ROC curve)
        plt.figure(figsize=(7,7));
        train_TPR = train_TP / (train_TP + train_FN)
        train_FPR = train_FP / (train_FP + train_TN)
        train_AUC = auc(train_FPR, train_TPR)
        plt.plot(train_FPR, train_TPR, '-', color='lime', label = "Training Set (Area = {0:.3f})".format(train_AUC))
        test_TPR = test_TP / (test_TP + test_FN)
        test_FPR = test_FP / (test_FP + test_TN)
        test_AUC = auc(test_FPR, test_TPR)
        plt.plot(test_FPR, test_TPR, '-', color='magenta', label = "Test Set (Area = {0:.3f})".format(test_AUC))

        # make the plot readable
        plt.xlabel('False Positive Rate (Background Efficiency)',fontsize=12);
        plt.ylabel('True Positive Rate (Signal Efficiency)',fontsize=12);
        plt.legend(frameon=False);
        if savefig:
            plt.savefig(out_dir + "TPR_FPR.pdf")
            plt.savefig(out_dir + "TPR_FPR.png")
        if plot:
            plt.draw()
        else:
            plt.close()
        plt.figure(figsize = (10,10))
        xgb.plot_importance(self.booster, grid=False, max_num_features=None);
        plt.gcf().subplots_adjust(left=0.4)
        if savefig:
            plt.savefig(out_dir + "Feature_Importance.pdf")
            plt.savefig(out_dir + "Feature_Importance.png")
        if plot:
            plt.draw()
        else:
            plt.close()
        os.makedirs(out_dir + "/histograms", exist_ok=True)
        labels = self.BDT_features[:(len(self.BDT_features)-1)]
        for label in labels:
            gen_hist(self.full_data, label, out_dir, savefig=savefig, plot=False)
            
        evals = self.evals_result
        plt.figure("logloss", figsize=(7,7))
        train_errors = evals["train"]["logloss"]
        test_errors = evals["test"]["logloss"]
        iterations = np.arange(1, len(test_errors)+1)
        plt.plot(iterations, train_errors, label="train")
        plt.plot(iterations, test_errors, label="test")
        plt.legend()
        plt.xlabel("Training Rounds (num_trees)")
        plt.ylabel("Log Loss")
        plt.title("{} Log Loss Plot".format(self.label))
        if savefig:
            plt.savefig(out_dir + "Logloss.pdf")
            plt.savefig(out_dir + "Logloss.png")
        if plot:
            plt.draw()
        else:
            plt.close()
            
        plt.figure("error", figsize=(7,7))
        train_errors = evals["train"]["error"]
        test_errors = evals["test"]["error"]
        iterations = np.arange(1, len(test_errors)+1)
        plt.plot(iterations, train_errors, label="train")
        plt.plot(iterations, test_errors, label="test")
        plt.legend()
        plt.xlabel("Training Rounds (num_trees)")
        plt.ylabel("Error")
        plt.title("{} Error Plot".format(self.label))
        #plt.yscale("log")
        if savefig:
            plt.savefig(out_dir + "Error.pdf")
            plt.savefig(out_dir + "Error.png")
        if plot:
            plt.draw()
        else:
            plt.close()
            
        plt.figure("rmse", figsize=(7,7))
        train_errors = evals["train"]["rmse"]
        test_errors = evals["test"]["rmse"]
        iterations = np.arange(1, len(test_errors)+1)
        plt.plot(iterations, train_errors, label="train")
        plt.plot(iterations, test_errors, label="test")
        plt.legend()
        plt.xlabel("Training Rounds (num_trees)")
        plt.ylabel("Root Mean Square Error")
        plt.title("{} RMSE Plot".format(self.label))
        #plt.yscale("log")
        if savefig:
            plt.savefig(out_dir + "RMSE.pdf")
            plt.savefig(out_dir + "RMSE.png")
        if plot:
            plt.draw()
        else:
            plt.close()

    def make_roc(self, signal=None, other_boosters=[], title="ROC Curves"):
        if signal==None:
            cat_dict = self.category_dict
        elif signal == "HCT":
            cat_dict = self.HCT_dict
        elif signal == "HUT":
            cat_dict = self.HUT_dict
            
        plt.figure(figsize=(7,7));
        data = cat_dict["all"]["data"]
        predictions = cat_dict["all"]["prediction"]
        FPR, TPR, thresholds = roc_curve((data.Label=="s").astype(int).to_numpy(), predictions, sample_weight=data.Weight.to_numpy())
        AUC=np.trapz(TPR, FPR)
        #AUC = roc_auc_score(data.Label=='s', predictions, sample_weight=data.Weight)
        ax = plt.gca()
        ax.plot(FPR, TPR, '-', label = "{0} (AUC={1:.3f})".format(self.label, AUC))
        for other in other_boosters:
            data = other.category_dict["all"]["data"]
            predictions = other.category_dict["all"]["prediction"]
            FPR, TPR, thresholds = roc_curve((data.Label=="s").astype(int).to_numpy(), predictions, sample_weight=data.Weight.to_numpy())
            AUC=np.trapz(TPR, FPR)
            ax.plot(FPR, TPR, '-', label = "{0} (AUC={1:.3f})".format(other.label, AUC))
        ax.legend()
        ax.set_xlabel('False Positive Rate (Background Efficiency)',fontsize=12);
        ax.set_ylabel('True Positive Rate (Signal Efficiency)',fontsize=12);
        ax.set_title(title)
        plt.draw()
        
    def make_category_dict(self, directories, sig_name=None, background="all", data_driven=False, from_pandas=False, systematics=True, flag_store_dict=True):
        sig_df = pd.DataFrame()
        fakes_df = pd.DataFrame()
        flips_df = pd.DataFrame()
        rares_df = pd.DataFrame()
        all_df = pd.DataFrame()
        background_df = pd.DataFrame()
        if data_driven and not from_pandas:
            for d in directories:
                if sig_name==None:
                    for s in ["signal_tch", "signal_tuh"]:
                        sig_df = pd.concat([sig_df, load_category(s, d, BDT_features=self.BDT_features, systematics=systematics)], axis=0)
                        all_df = pd.concat([all_df, load_category(s, d, BDT_features=self.BDT_features, systematics=systematics)], axis=0)
                elif sig_name=="HCT":
                    sig_df = pd.concat([sig_df, load_category("signal_tch", d, BDT_features=self.BDT_features, systematics=systematics)])
                    all_df = pd.concat([all_df, load_category("signal_tch", d, BDT_features=self.BDT_features, systematics=systematics)])
                elif sig_name=="HUT":
                    sig_df = pd.concat([sig_df, load_category("signal_tuh", d, BDT_features=self.BDT_features, systematics=systematics)])
                    all_df = pd.concat([all_df, load_category("signal_tuh", d, BDT_features=self.BDT_features, systematics=systematics)])
                fakes_df = pd.concat([fakes_df, load_category("data_fakes", d, BDT_features=self.BDT_features)]) #no systematics for fakes
                if (background=="all") or (background=="fakes"):
                    all_df = pd.concat([all_df, load_category("data_fakes", d, BDT_features=self.BDT_features)]) #no systematics for fakes
                flips_df = pd.concat([flips_df, load_category("data_flips", d, BDT_features=self.BDT_features)]) #no systematics for flips
                rares_df = pd.concat([rares_df, load_category("rares", d, BDT_features=self.BDT_features, systematics=systematics)])
                if (background=="all") or (background=="flips"):
                    all_df = pd.concat([all_df, load_category("data_flips", d, BDT_features=self.BDT_features)]) #no systematics for flips
                    all_df = pd.concat([all_df, load_category("rares", d, BDT_features=self.BDT_features, systematics=systematics)])
        elif not from_pandas:
            for d in directories:
                if sig_name==None:
                    for s in ["signal_tch", "signal_tuh"]:
                        sig_df = pd.concat([sig_df, load_category(s, d, BDT_features=self.BDT_features, systematics=systematics)], axis=0)
                        all_df = pd.concat([all_df, load_category(s, d, BDT_features=self.BDT_features, systematics=systematics)], axis=0)
                elif sig_name=="HCT":
                    sig_df = pd.concat([sig_df, load_category("signal_tch", d, BDT_features=self.BDT_features, systematics=systematics)])
                    all_df = pd.concat([all_df, load_category("signal_tch", d, BDT_features=self.BDT_features, systematics=systematics)])
                elif sig_name=="HUT":
                    sig_df = pd.concat([sig_df, load_category("signal_tuh", d, BDT_features=self.BDT_features, systematics=systematics)])
                    all_df = pd.concat([all_df, load_category("signal_tuh", d, BDT_features=self.BDT_features, systematics=systematics)])
                fakes_df = pd.concat([fakes_df, load_category("fakes_mc", d, BDT_features=self.BDT_features)]) #no systematics for fakes
                if (background=="all") or (background=="fakes"):
                    all_df = pd.concat([all_df, load_category("fakes_mc", d, BDT_features=self.BDT_features)]) #no systematics for fakes
                flips_df = pd.concat([flips_df, load_category("flips_mc", d, BDT_features=self.BDT_features)]) #no systematics for flips
                rares_df = pd.concat([rares_df, load_category("rares", d, BDT_features=self.BDT_features, systematics=systematics)])
                if (background=="all") or (background=="flips"):
                    all_df = pd.concat([all_df, load_category("flips_mc", d, BDT_features=self.BDT_features)]) #no systematics for flips
                    all_df = pd.concat([all_df, load_category("rares", d, BDT_features=self.BDT_features, systematics=systematics)])
        elif from_pandas:

            if sig_name == "HCT":
                sig_label = -2
            elif sig_name == "HUT":
                sig_label = -1
            for d in directories:
                baby = pd.read_hdf(d)
                sig = pd.DataFrame()
                fakes = pd.DataFrame()
                flips = pd.DataFrame()
                rares = pd.DataFrame()
                for f in self.BDT_features:
                    sig[f] = baby[baby.label==sig_label][translate[f]]
                    fakes[f] = baby[baby.label>=3][translate[f]]
                    flips[f] = baby[baby.label==2][translate[f]]
                    rares[f] = baby[baby.label==1][translate[f]]
                sig_df = pd.concat([sig_df, sig], axis=0)
                fakes_df = pd.concat([fakes_df, fakes])
                flips_df = pd.concat([flips_df, flips])
                rares_df = pd.concat([rares_df, rares])

            sig_df["Label"] = "s"
            sig_df["Weight"] *= 100
            fakes_df["Label"] = "b"
            flips_df["Label"] = "b"
            rares_df["Label"] = "b"
            if (background=="all"):
                all_df = pd.concat([sig_df, fakes_df, flips_df, rares_df], axis=0)
            elif (background=="fakes"):
                all_df = pd.concat([sig_df, fakes_df], axis=0)
            elif (background=="flips"):
                all_df = pd.concat([sig_df, flips_df, rares_df], axis=0)
            #end pandas df load
        sig_pred = self.booster.predict(make_dmatrix(sig_df, self.BDT_features))
        fakes_pred = self.booster.predict(make_dmatrix(fakes_df, self.BDT_features))
        flips_pred = self.booster.predict(make_dmatrix(flips_df, self.BDT_features))
        rares_pred = self.booster.predict(make_dmatrix(rares_df, self.BDT_features))
        all_pred = self.booster.predict(make_dmatrix(all_df, self.BDT_features))
        cat_dict = {"signal":{"data":sig_df, "prediction":sig_pred}, "fakes":{"data":fakes_df, "prediction":fakes_pred},
                    "flips":{"data":flips_df, "prediction":flips_pred}, "rares":{"data":rares_df, "prediction":rares_pred},
                    "all":{"data":all_df, "prediction":all_pred}}
        if flag_store_dict:
            if sig_name == None:
                self.category_dict = cat_dict
                return self.category_dict
            elif sig_name == "HCT":
                self.HCT_dict = cat_dict
                return self.HCT_dict
            elif sig_name == "HUT":
                self.HUT_dict = cat_dict
                return self.HUT_dict
        else:
            return cat_dict
    
    def plot_categories(self, plot=True, savefig=True):
        out_dir = "{0}/{1}/{2}/".format(self.out_base_dir, self.label, self.booster_label)
        BDT_signal = self.category_dict["signal"]["data"]
        BDT_fakes = self.category_dict["fakes"]["data"]
        BDT_flips = self.category_dict["flips"]["data"]
        BDT_rares = self.category_dict["rares"]["data"]
        feature_names = self.BDT_features.copy()
        feature_names.pop(-1)#BDT_fakes.columns[1:-2]
        BDT_fakes['Label'] = BDT_fakes.Label.astype('category')
        BDT_flips['Label'] = BDT_flips.Label.astype('category')
        BDT_rares['Label'] = BDT_rares.Label.astype('category')
        BDT_signal['Label'] = BDT_signal.Label.astype('category')
        fakes_predictions = self.category_dict["fakes"]["prediction"]
        flips_predictions = self.category_dict["flips"]["prediction"]
        rares_predictions = self.category_dict["rares"]["prediction"]
        signal_predictions = self.category_dict["signal"]["prediction"]
        plt.figure(figsize=(7,7));
        #tmp_bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.50, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75, 0.775, 0.8, 0.85, 1.0])
        tmp_bins = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.hist(fakes_predictions,bins=tmp_bins,histtype='step',color='#d7191c',label='fakes', weights=BDT_fakes.Weight)#, density=True);
        plt.hist(flips_predictions,bins=tmp_bins,histtype='step',color='#2c7bb6',label='flips', weights=BDT_flips.Weight)#, density=True);
        plt.hist(rares_predictions,bins=tmp_bins,histtype='step',color='#fdae61',label='rares', weights=BDT_rares.Weight)#, density=True);
        plt.hist(signal_predictions,bins=tmp_bins,histtype='step',color='black',label='signal', weights=BDT_signal.Weight/100)#, density=True);

        # make the plot readable
        plt.xlabel('Prediction from BDT',fontsize=12);
        plt.ylabel('Yield',fontsize=12);
        plt.legend(frameon=False);
        plt.title(self.label)
        if savefig:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(out_dir + "Background_Categories.pdf")
            plt.savefig(out_dir + "Background_Categories.png")
        if plot:
            plt.draw()
        else:
            plt.close()
            
    def plot_SB_ratio(self, cat_dict, sig_pred, fakes_pred, flips_pred, rares_pred, bins):
        out_dir = "{0}/{1}/{2}/".format(self.out_base_dir, self.label, self.booster_label)
        sig_weight = np.array(cat_dict["signal"]["data"].Weight)
        back_weight = np.concatenate((cat_dict["fakes"]["data"].Weight, cat_dict["flips"]["data"].Weight, cat_dict["rares"]["data"].Weight))
        back_pred = np.concatenate((fakes_pred, flips_pred, rares_pred))
        sig_hist = Hist1D(sig_pred, bins=bins, weights=sig_weight/100)
        back_hist = Hist1D(back_pred, bins=bins, weights=back_weight)
        tot_hist = sig_hist + back_hist
        denom_hist = Hist1D.from_bincounts(np.sqrt(np.array(tot_hist.counts)), bins=tot_hist.edges)
        result = sig_hist.divide(denom_hist)
        plt.figure("comparision", figsize=(7,7))
        result.plot(ax=plt.gca())
        plt.title(r'$S/\sqrt{S + B}$')
        plt.xlabel("BDT Score (after QT)")
        plt.ylabel("Ratio")
        plt.savefig(out_dir + "SB_ratio.pdf")
        plt.savefig(out_dir + "SB_ratio.png")
        plt.close()
        
    def plot_datacard_bins(self, cat_dict, sig_pred, fakes_pred, flips_pred, rares_pred, bins, year):
        out_dir = "{0}/{1}/{2}/datacard_plots/".format(self.out_base_dir, self.label, self.booster_label)
        os.makedirs(out_dir, exist_ok=True)
        sig_weight = np.array(cat_dict["signal"]["data"].Weight)
        plt.figure("dc_categories_{}".format(year), figsize=(7,7))
        plt.hist(fakes_pred,bins=bins,histtype='step',color='#d7191c',label='fakes', weights=cat_dict["fakes"]["data"].Weight)
        plt.hist(flips_pred,bins=bins,histtype='step',color='#2c7bb6',label='flips', weights=cat_dict["flips"]["data"].Weight)
        plt.hist(rares_pred,bins=bins,histtype='step',color='#fdae61',label='rares', weights=cat_dict["rares"]["data"].Weight)
        plt.hist(sig_pred,bins=bins,histtype='step',color='black',label='signal', weights=cat_dict["signal"]["data"].Weight/100)
        plt.title('Datacard Bins ({})'.format(year))
        plt.legend()
        plt.xlabel("BDT Score (after QT)")
        plt.savefig(out_dir + "DC_bins_{}.pdf".format(year))
        plt.savefig(out_dir + "DC_bins_{}.png".format(year))
        plt.close()
        
        plt.figure("fakes_stats_{}".format(year), figsize=(7,7))
        fakes_hist = Hist1D(fakes_pred, bins=bins)
        #plt.hist(fakes_pred,bins=bins,histtype='step',color='#d7191c',label='fakes')
        fakes_hist.plot(ax=plt.gca(), errors=True, color = '#d7191c')
        plt.title('Fakes CR Stats ({})'.format(year))
        plt.xlabel("BDT Score (after QT)")
        plt.ylabel("Number of Events")
        plt.savefig(out_dir + "Fakes_CRStats_{}.pdf".format(year))
        plt.savefig(out_dir + "Fakes_CRStats_{}.png".format(year))
        plt.close()
        
        plt.figure("flips_stats_{}".format(year), figsize=(7,7))
        flips_hist = Hist1D(flips_pred, bins=bins)
        flips_hist.plot(ax=plt.gca(), errors=True, color = '#2c7bb6')
        plt.title('Flips CR Stats ({})'.format(year))
        plt.ylabel("Number of Events")
        plt.xlabel("BDT Score (after QT)")
        plt.savefig(out_dir + "Flips_CRStats_{}.pdf".format(year))
        plt.savefig(out_dir + "Flips_CRStats_{}.png".format(year))
        plt.close()
        
    def get_bin_yield(self, category, bins, bin_idx, cat_dict, quantile_transformer=None):
        tmp_weights = cat_dict[category]["data"].Weight.copy()
        if category=="signal":
            tmp_weights /= 100.
        if len(tmp_weights) > 0:
            if quantile_transformer==None:
                predictions = cat_dict[category]["prediction"] 
            else:
                predictions = quantile_transformer.transform(cat_dict[category]["prediction"].reshape(-1, 1)).flatten()
            digitized_prediction = np.digitize(predictions, bins)
            tmp_yield = np.sum(tmp_weights[digitized_prediction==bin_idx])
            tmp_BDT_hist = Hist1D(predictions, bins=bins, weights=tmp_weights, overflow=False)
            tmp_error = tmp_BDT_hist.errors[bin_idx-1]
        else: #special case (trilep has no flips)
            tmp_yield = 0
            tmp_error = 0.
        yield_name = "bin_{0}_{1}".format(bin_idx-1, category)
        return yield_name, tmp_yield, tmp_error
    
    def get_bin_CRStats(self, bins, bin_idx, cat_dict, quantile_transformer=None):
        weights = cat_dict["fakes"]["data"].Weight.copy()
        if quantile_transformer==None:
            predictions = cat_dict["fakes"]["prediction"]
        else:
            predictions = quantile_transformer.transform(cat_dict["fakes"]["prediction"].reshape(-1, 1)).flatten()
        digitized_prediction = np.digitize(predictions, bins)
        tmp_yield = np.sum(weights[digitized_prediction==bin_idx])
        tmp_counts = np.sum((digitized_prediction==bin_idx))
        return tmp_yield, tmp_counts
    
    def get_bin_systematic_yield(self, category, systematic, bins, bin_idx, cat_dict, quantile_transformer=None, JES_cds=()): 
        if (systematic=="JES") and (len(JES_cds)==2):
            weights_up = JES_cds[0][category]["data"]["Weight"].copy()
            weights_down = JES_cds[1][category]["data"]["Weight"].copy()
        else:
            weights_up = cat_dict[category]["data"]["Weight_{}_up".format(systematic)].copy()
            weights_down = cat_dict[category]["data"]["Weight_{}_down".format(systematic)].copy()
        if category=="signal":
            weights_up /= 100.
            weights_down /= 100.
        if len(weights_up) > 0:
            if (systematic=="JES") and (len(JES_cds)==2):
                if quantile_transformer==None:
                    predictions_up = JES_cds[0][category]["prediction"]
                    predictions_down = JES_cds[1][category]["prediction"]
                else:
                    predictions_up = quantile_transformer.transform(JES_cds[0][category]["prediction"].reshape(-1, 1)).flatten()
                    predictions_down = quantile_transformer.transform(JES_cds[1][category]["prediction"].reshape(-1, 1)).flatten()
                digitized_prediction_up = np.digitize(predictions_up, bins)
                digitized_prediction_down = np.digitize(predictions_down, bins)
                up_yield = np.sum(weights_up[digitized_prediction_up==bin_idx])
                down_yield = np.sum(weights_down[digitized_prediction_down==bin_idx])
            else:
                if quantile_transformer==None:
                    predictions = cat_dict[category]["prediction"] ##fix the prediction
                else:
                    predictions = quantile_transformer.transform(cat_dict[category]["prediction"].reshape(-1, 1)).flatten()
                digitized_prediction = np.digitize(predictions, bins)
                up_yield = np.sum(weights_up[digitized_prediction==bin_idx])
                down_yield = np.sum(weights_down[digitized_prediction==bin_idx])
        else:
            up_yield = 0
            down_yield = 0
        return (up_yield, down_yield)

    def get_QT_bins(self, signal_name, year, directories, bins, data_driven=False):
        if signal_name == "HCT":
            cat_dict = self.make_category_dict(directories, "HCT", background="all", data_driven=data_driven, from_pandas=False, systematics=False, flag_store_dict=False)
        elif signal_name == "HUT":
            cat_dict = self.make_category_dict(directories, "HUT", background="all", data_driven=data_driven, from_pandas=False, systematics=False, flag_store_dict=False)

        signal_predictions = cat_dict["signal"]["prediction"]
        qt = QuantileTransformer(output_distribution="uniform")
        qt.fit(signal_predictions.reshape(-1, 1))
        transformed_bins = qt.inverse_transform(bins.reshape(-1, 1))
        return transformed_bins[:,0]
    
    def make_BDT_test_csv(self, test_file, output_dir):
        output_dir = "{0}/{1}/{2}/".format(self.out_base_dir, self.label, self.booster_label)
        #booster = xgb.Booster() # init model
        #self.booster.load_model(booster_path) # load data
        test_df = pd.read_csv(test_file)
        results = self.booster.predict(make_dmatrix(test_df, self.BDT_features))
        test_df["result"] = results
        test_df = test_df.drop(labels=["Weight", "Label", "Category"], axis=1)
        test_df.to_csv("{}/python_test_results_{}.csv".format(output_dir, self.label), index=False)
        print(results)
    
    def gen_datacard(self, signal_name, year, directories, quantile_transform=True, data_driven=False, plot=True, BDT_bins=np.linspace(0, 1, 21),
                     flag_tmp_directory=False, dir_label="tmp", from_pandas=False, systematics=False, JES_dirs=()):
        yield_dict = {}
        if systematics:
            systematics_yields = {}
        else:
            systematics_yields = None
        if signal_name == "HCT":
            cat_dict = self.make_category_dict(directories, "HCT", background="all", data_driven=data_driven, from_pandas=from_pandas, systematics=systematics, flag_store_dict=False)
            if len(JES_dirs)==2:
                JES_up_cd = self.make_category_dict(JES_dirs[0], "HCT", background="all", data_driven=data_driven, from_pandas=from_pandas, systematics=systematics, flag_store_dict=False)
                JES_down_cd = self.make_category_dict(JES_dirs[1], "HCT", background="all", data_driven=data_driven, from_pandas=from_pandas, systematics=systematics, flag_store_dict=False)
        elif signal_name == "HUT":
            cat_dict = self.make_category_dict(directories, "HUT", background="all", data_driven=data_driven, from_pandas=from_pandas, systematics=systematics, flag_store_dict=False)
            if (len(JES_dirs)==2 and systematics):
                JES_up_cd = self.make_category_dict(JES_dirs[0], "HUT", background="all", data_driven=data_driven, from_pandas=from_pandas, systematics=systematics, flag_store_dict=False)
                JES_down_cd = self.make_category_dict(JES_dirs[1], "HUT", background="all", data_driven=data_driven, from_pandas=from_pandas, systematics=systematics, flag_store_dict=False)
        out_dir = "{0}/{1}/datacards/".format(self.out_base_dir, self.label)
        if flag_tmp_directory:
            out_dir += "tmp/{}/".format(dir_label)
        BDT_signal = cat_dict["signal"]["data"]
        BDT_fakes = cat_dict["fakes"]["data"]
        BDT_flips = cat_dict["flips"]["data"]
        BDT_rares = cat_dict["rares"]["data"]
        #BDT_bins = np.linspace(0, 1, 20)
        feature_names = self.BDT_features.copy()
        feature_names.pop(-1)
        fakes_weights = BDT_fakes.Weight 
        flips_weights = BDT_flips.Weight
        rares_weights = BDT_rares.Weight 
        signal_weights = BDT_signal.Weight / 100.0
        if quantile_transform == True:
            signal_predictions = cat_dict["signal"]["prediction"]
            qt = QuantileTransformer(output_distribution="uniform")
            qt.fit(signal_predictions.reshape(-1, 1))
            signal_predictions = qt.transform(signal_predictions.reshape(-1, 1)).flatten()
            fakes_predictions = qt.transform(cat_dict["fakes"]["prediction"].reshape(-1, 1)).flatten()
            if len(flips_weights) > 0:
                flips_predictions = qt.transform(cat_dict["flips"]["prediction"].reshape(-1, 1)).flatten()
            else:
                flips_predictions = np.array([0.])
            rares_predictions = qt.transform(cat_dict["rares"]["prediction"].reshape(-1, 1)).flatten()
            if plot:
                self.plot_SB_ratio(cat_dict, signal_predictions, fakes_predictions, flips_predictions, rares_predictions, BDT_bins)
                self.plot_datacard_bins(cat_dict, signal_predictions, fakes_predictions, flips_predictions, rares_predictions, BDT_bins, year)
        else:      
            qt=None
        for b in range(1, len(BDT_bins)): #get yields of each bin
            background_sum = 0
            for category in ["signal", "fakes", "flips", "rares"]:
                (yield_name, tmp_yield, tmp_error) = self.get_bin_yield(category, BDT_bins, b, cat_dict, qt)
                if category != "signal":
                    background_sum += tmp_yield
                if ((category=="signal") or (category=="rares")) and systematics: #fill systematic uncertainties for signal and rares
                    for sys in ["LepSF","PU","Trigger","bTag", "JES"]: ##TODO: add JES
                        if (sys=="JES") and (len(JES_dirs)==2):
                            sys_yields = self.get_bin_systematic_yield(category, sys, BDT_bins, b, cat_dict, qt, (JES_up_cd, JES_down_cd))
                        else:
                            sys_yields = self.get_bin_systematic_yield(category, sys, BDT_bins, b, cat_dict, qt)
                        systematics_yields["{0}_{1}_up".format(yield_name, sys)] = sys_yields[0]
                        systematics_yields["{0}_{1}_down".format(yield_name, sys)] = sys_yields[1]
                elif (category=="fakes") and systematics:
                    stat_name = "fkStat{0}_{1}".format(str(year)[-2:], b-1)
                    sys_yields = self.get_bin_CRStats(BDT_bins, b, cat_dict, qt)
                    systematics_yields[stat_name] = sys_yields[1]
                    #yield_dict[yield_name + "_error"] = sys_yields[0] / sys_yields[1]
                if tmp_yield > 0:
                    yield_dict[yield_name] = tmp_yield
                    if (category=="fakes") and systematics:
                        yield_dict[yield_name + "_error"] = sys_yields[0] / sys_yields[1] #fakes have their own CR error 
                    else:
                        yield_dict[yield_name+"_error"] = tmp_error
                elif tmp_yield <=0:
                    yield_dict[yield_name] = 0.01
                    yield_dict[yield_name+"_error"] = 0.0
            yield_dict["bin_{0}_Total_Background".format(b-1)] = background_sum
        
        output_path = out_dir + signal_name + "/"
        os.makedirs(output_path, exist_ok=True)
        if quantile_transform == True:
            label = "QT"
        else:
            label = ""
        postProcessing.makeCards.make_BDT_datacard(yield_dict, BDT_bins, signal_name, output_path, label, year, systematics_yields=systematics_yields, flag_plot=plot)
        
    def plot_response(self, plot=True, savefig=False, label="all"):
        #plot the BDT response (correlation of BDT score with feature variables)
        #also plot the feature importance for comparison
        corr_df = self.category_dict[label]["data"].copy()
        corr_df["BDT_response"] = self.category_dict[label]["prediction"]
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        response = corr_df.corr().BDT_response
        #response_labels =  corr_df.select_dtypes(['number']).columns
        response_labels = self.booster.get_fscore().keys()
        bdt_response = []
        for l in response_labels:
            if np.isnan(response[l]):
                bdt_response.append(0.0)
            else:
                bdt_response.append(response[l])
        bdt_response = np.array(bdt_response)
        sorted_args = np.argsort(np.abs(bdt_response))[::-1]
        bdt_fscore = np.array([self.booster.get_fscore()[l] for l in response_labels])[sorted_args]
        fscore_scale = np.max(bdt_fscore)
        bdt_fscore = bdt_fscore / fscore_scale
        bdt_fscore_plot = ax.bar(range(len(bdt_response)), bdt_fscore, label = "BDT Feature Importance", facecolor=(0,0,0,0), edgecolor='blue')
        bdt_response_plot = ax.bar(range(len(bdt_response)), bdt_response[sorted_args], label = "BDT Response", facecolor=(0,0,0,0), edgecolor='orange')

        ax.set_xticks(range(len(response_labels)))
        ax.set_xticklabels(np.array(list(response_labels))[sorted_args], rotation=90, fontdict={'fontsize':12})
        ax.legend()
        ax.set_title('{} {} BDT Response vs Feature Importance'.format(self.label, label), fontsize=16)
        plt.gcf().subplots_adjust(bottom=0.3)
        if savefig:
            out_dir = "{0}/{1}/{2}/".format(self.out_base_dir, self.label, self.booster_label)
            plt.savefig(out_dir + "BDT_{}_Response.pdf".format(label))
            plt.savefig(out_dir + "BDT_{}_Response.png".format(label))
        if plot:
            plt.draw()
        else:
            plt.close()
    
    def gen_BDT_and_plot(self, load_BDT=True, optimize=True, retrain=True, flag_save_booster=True, plot=True):
        if optimize:
            self.optimize_booster()
        elif load_BDT:
            self.load_booster_params()
            self.set_booster_label()
        self.gen_BDT(flag_load=(not retrain), flag_save_booster=flag_save_booster) #load=True
        self.get_predictions()
        if plot:
            self.gen_prediction_plots(savefig=True, plot=False)
            
    def fill_dicts(self, directories, background="all", data_driven=False):
        self.make_category_dict(directories, background=background, data_driven=data_driven)
        self.make_category_dict(directories, "HCT", background=background, data_driven=data_driven)
        self.make_category_dict(directories, "HUT", background=background, data_driven=data_driven)
        
    def gen_datacards(self, directory, year, quantile_transform=True, data_driven=True, BDT_bins=np.linspace(0, 1, 21), flag_tmp_directory=False, plot=True, from_pandas=False, systematics=True, JES_dirs=(), flag_plot=False):
        self.gen_datacard("HCT", year, directory, quantile_transform, data_driven, BDT_bins=BDT_bins, flag_tmp_directory=flag_tmp_directory,
                          plot=plot, from_pandas=from_pandas, systematics=systematics, JES_dirs=JES_dirs, flag_plot=flag_plot)
        self.gen_datacard("HUT", year, directory, quantile_transform, data_driven, BDT_bins=BDT_bins, flag_tmp_directory=flag_tmp_directory,
                          plot=plot, from_pandas=from_pandas, systematics=systematics, JES_dirs=JES_dirs, flag_plot=flag_plot)
        
    def load_custom_params(self, param, dirs): #given a specified set of hyperparameters as a dictionary, train a new BDT
        self.set_booster_label()
        self.load_booster_params()
        print(self.booster_params)
        self.booster_params = param
        self.gen_BDT(False)
        self.get_predictions()
        self.gen_prediction_plots(savefig=True, plot=True)
        self.fill_dicts(dirs)
        self.plot_categories()
        self.gen_datacards()
        self.make_roc()
        
def compare_S_B_ratio(BDTs, directories="", fill_categories=True):
    #compare the ratios
    tmp_bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.50, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75, 0.775, 0.8, 0.85, 1.0])
    ratio_hists = {}
    for b in BDTs:
        if fill_categories:
            b.make_category_dict(directories)
        combined_background = np.concatenate([b.category_dict[c]["prediction"] for c in ["fakes", "flips", "rares"]])
        combined_background_weight = np.concatenate([b.category_dict[c]["data"].Weight for c in ["fakes", "flips", "rares"]])
        combined_signal_weight = b.category_dict["signal"]["data"].Weight
        combined_signal_prediction_hist = Hist1D(b.category_dict["signal"]["prediction"], bins=tmp_bins, weights=combined_signal_weight/100)
        combined_background_prediction_hist = Hist1D(combined_background, bins=tmp_bins, weights=combined_background_weight)
        plt.figure("sig/back combined: {}".format(b.label), figsize=(7,7))
        combined_signal_prediction_hist.plot(ax=plt.gca(), label="Signal")
        combined_background_prediction_hist.plot(ax=plt.gca(), label="Background")
        plt.title(b.label)
        plt.xlabel("BDT Score")
        plt.ylabel("Weighted Yield")
        plt.legend()
        ratio_hists[b.label] = get_s_b_ratio(combined_signal_prediction_hist, combined_background_prediction_hist)
        
    plt.figure("comparison", figsize = (7,7))
    for b in BDTs:    
        combined_ratio = ratio_hists[b.label]
        combined_ratio.plot(ax=plt.gca(), label = "{} Ratio".format(b.label))

    plt.title(r'BDT $S/\sqrt{S + B}$ Comparison')
    plt.xlabel("BDT Score")
    plt.ylabel("Ratio")
    plt.legend()
    plt.draw()
