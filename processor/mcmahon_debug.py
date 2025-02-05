from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import awkward as ak

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection

import numpy as np
import pandas as pd
from yahist import Hist1D, Hist2D

# this is all very bad practice
from Tools.objects import *
from Tools.basic_objects import *
from Tools.cutflow import *
from Tools.config_helpers import *
from Tools.triggers import *
from Tools.btag_scalefactors import *
from Tools.lepton_scalefactors import *
from Tools.helpers import mt
from Tools.fake_rate import fake_rate
from Tools.SS_selection import SS_selection
import production.weights
# the below command will change to .from_root in coffea v0.7.0
# events = NanoEventsFactory.from_root('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.3/ProjectMetis_TTWJetsToLNuEWK_5f_NLO_RunIIAutumn18_NANO_v2/nanoSkim_1.root', schemaclass=NanoAODSchema).events()

# events = NanoEventsFactory.from_root('root://xcache-redirector.t2.ucsd.edu:2040//store/mc/RunIIAutumn18NanoAODv7/QCD_Pt-120to170_MuEnrichedPt5_TuneCP5_13TeV_pythia8/NANOAODSIM/Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/70000/DE335891-829A-B943-99BE-E5A179F5F3EB.root', schemaclass=NanoAODSchema).events()
events = NanoEventsFactory.from_root('/nfs-7/userdata/ksalyer/fcnc/fcnc_v10_SRonly_11june2021/2018/dyjets_m10-50.root', schemaclass=NanoAODSchema).events()
#
year = 2018

events = events[ak.num(events.Jet)>0] #corrects for rare case where there isn't a single jet 

# we can use a very loose preselection to filter the events. nothing is done with this presel, though
presel = ak.num(events.Jet)>=2

ev = events[presel]
##Jets
Jets = events.Jet

## MET -> can switch to puppi MET
met_pt  = ev.MET.pt
met_phi = ev.MET.phi

 ### For FCNC, we want electron -> tightTTH
ele_t = Collections(ev, "Electron", "tightFCNC", year=year).get()
ele_l = Collections(ev, "Electron", "fakeableFCNC", year=year).get()    
mu_t  = Collections(ev, "Muon", "tightFCNC", year=year).get()
mu_l  = Collections(ev, "Muon", "fakeableFCNC", year=year).get()

dataset = 'dy'
rares_dataset = ['ttw', 'www', 'wzz', 'wz', 'tg', 'wzg', 'tth_nobb', 'ttg_dilep', 'tttj', 'tttt', 'tttw', 
                 'tthh', 'vh_nobb', 'tzq', 'ttwh', 'ttg_1lep', 'wwz', 'wwg', 'ttwz', 'zz', 'ttww', 'wg', 
                 'ttz_m10', 'qqww', 'zzz', 'ggh', 'ttzh', 'ttz_m1-10', 'tw_dilep', 'ttzz']
flips_dataset = ['dy']
if dataset in rares_dataset:
    ele_t = ele_t[((ele_t.genPartFlav==1)|(ele_t.genPartFlav==15))]
    ele_l = ele_l[((ele_l.genPartFlav==1)|(ele_l.genPartFlav==15))]
    mu_t = mu_t[((mu_t.genPartFlav==1)|(mu_t.genPartFlav==15))]
    mu_l = mu_l[((mu_l.genPartFlav==1)|(mu_l.genPartFlav==15))]


#SS preselection 
lepton  = ak.concatenate([mu_l, ele_l], axis=1)
tight_lepton  = ak.concatenate([mu_t, ele_t], axis=1)
sorted_index_nofilter = ak.argsort(tight_lepton.pt, axis=-1, ascending=False)
sorted_lep_nofilter = tight_lepton[sorted_index_nofilter]
leadlep_nofilter = sorted_lep_nofilter[:,0:1]
subleadlep_nofilter = sorted_lep_nofilter[:,1:2]

#M(ee) > 12
diele_mass = choose(ele_t, 2).mass

#clean jets :
# we want at least two jets that are outside of the lepton jets by deltaR > 0.4
jets = getJets(ev, maxEta=2.4, minPt=40, pt_var='pt')
jets_for_btag = getJets(ev, maxEta=2.5, minPt=25, pt_var='pt')
#jet_sel = (ak.num(jets[~(match(jets, ele_l, deltaRCut=0.4) | match(jets, mu_l, deltaRCut=0.4))])>=2)
btag = getBTagsDeepFlavB(jets_for_btag, year=year)


    
selection = PackedSelection()
selection.add("njets", (ak.num(jets[~(match(jets, tight_lepton, deltaRCut=0.4))]) >= 2))
selection.add("nlep", (ak.num(lepton, axis=1) == 2))
selection.add("nlep_tight", (ak.num(tight_lepton, axis=1) == 2))
selection.add("SS", (ak.sum(ak.concatenate([leadlep_nofilter.charge, subleadlep_nofilter.charge], axis=1), axis=1) != 0))
selection.add("nbtag", (ak.num(btag, axis=1) >= 0))
selection.add("M(ee)>12", ((ak.num(ele_t) < 2) | (ak.sum(diele_mass, axis=1) > 12.0))) #ak.sum here to flatten the diele_mass array
selection_reqs = ["njets", "nbtag", "nlep", "SS", "nlep_tight", "M(ee)>12"]

if dataset in flips_dataset:
    flip_evts = ((ak.sum(ev.GenPart.pdgId[ele_t.genPartIdx]==(-1 *ele_t.pdgId), axis=1) > 0))
    selection.add("flip", flip_evts)
    selection_reqs += ["flip"]
fcnc_reqs_d = { sel: True for sel in selection_reqs}
FCNC_sel = selection.require(**fcnc_reqs_d)
breakpoint()
#sorting
sorted_index = ak.argsort(lepton[FCNC_sel].pt, axis=-1, ascending=False)
sorted_pt = lepton[FCNC_sel].pt[sorted_index]
sorted_eta = lepton[FCNC_sel].eta[sorted_index]
sorted_phi = lepton[FCNC_sel].phi[sorted_index]
sorted_dxy = lepton[FCNC_sel].dxy[sorted_index]
sorted_dz = lepton[FCNC_sel].dz[sorted_index]
if (np.array(ak.num(jets[FCNC_sel])).any()==1): #if there is at least one event with a jet
    
    sorted_jet_index = ak.argsort(jets[FCNC_sel].pt, axis=-1, ascending=False)
    sorted_jet_pt = jets[FCNC_sel].pt[sorted_jet_index]
    #njets
    njets = ak.num(jets, axis=1)[FCNC_sel]
    most_forward_pt = ak.flatten(jets[FCNC_sel].pt[ak.singletons(ak.argmax(abs(jets[FCNC_sel].eta), axis=1))])
    leadjet_pt = ak.flatten(sorted_jet_pt[:,0:1])
    subleadjet_pt = ak.flatten(sorted_jet_pt[:,1:2])
    #this sometimes is not defined, so ak.firsts relpaces the empty arrays with None, then we can set all None to zero
    subsubleadjet_pt = ak.fill_none(ak.firsts(sorted_jet_pt[:,2:3]), 0)
else: #if there are no events with jets
    njets = np.zeros_like(FCNC_sel)
    most_forward_pt = np.zeros_like(FCNC_sel)
    leadjet_pt = np.zeros_like(FCNC_sel)
    subleadjet_pt = np.zeros_like(FCNC_sel)
    subsubleadjet_pt = np.zeros_like(FCNC_sel)

if (np.array(ak.num(btag[FCNC_sel])).any()==1): #if there is at least one event with a btag
    sorted_btag_index = ak.argsort(btag[FCNC_sel].pt, axis=-1, ascending=False)
    sorted_btag_pt = btag[FCNC_sel].pt[sorted_btag_index]
    #btags
    nbtag = ak.num(btag)[FCNC_sel]
    leadbtag_pt = sorted_btag_pt[:,0:1] #this sometimes is not defined (some of the arrays are empty)
    # ak.firsts() relpaces the empty arrays with None, then we can set all None to zero
    leadbtag_pt = ak.fill_none(ak.firsts(leadbtag_pt), 0)    

else:
    nbtag = np.zeros_like(FCNC_sel[FCNC_sel])
    leadbtag_pt = np.zeros_like(FCNC_sel[FCNC_sel])
    
    
leadlep_pt = ak.flatten(sorted_pt[:,0:1])
subleadlep_pt = ak.flatten(sorted_pt[:,1:2])
leadlep_eta = ak.flatten(sorted_eta[:,0:1])
subleadlep_eta = ak.flatten(sorted_eta[:,1:2])
leadlep_phi = ak.flatten(sorted_phi[:,0:1])
subleadlep_phi = ak.flatten(sorted_phi[:,1:2])
leadlep_dxy = ak.flatten(sorted_dxy[:,0:1])
subleadlep_dxy = ak.flatten(sorted_dxy[:,1:2])    
leadlep_dz = ak.flatten(sorted_dz[:,0:1])
subleadlep_dz = ak.flatten(sorted_dz[:,1:2])

sorted_lep = lepton[FCNC_sel][sorted_index]
leadlep = sorted_lep[:,0:1]
subleadlep = sorted_lep[:,1:2]
if (np.array(ak.num(leadlep)).any()==1) and (np.array(ak.num(subleadlep)).any()==1):
    leadlep_subleadlep_mass = ak.flatten((leadlep + subleadlep).mass)
    mt_leadlep_met = mt(leadlep_pt, leadlep_phi, MET_pt, MET_phi)
    mt_subleadlep_met = mt(subleadlep_pt, subleadlep_phi, MET_pt, MET_phi)
else:
    leadlep_subleadlep_mass = np.zeros_like(FCNC_sel[FCNC_sel])
    mt_leadlep_met = np.zeros_like(FCNC_sel[FCNC_sel])
    mt_subleadlep_met = np.zeros_like(FCNC_sel[FCNC_sel])
    
nelectron = ak.num(ele_l[FCNC_sel], axis=1)
MET_pt = ev[FCNC_sel].MET.pt
MET_phi = ev[FCNC_sel].MET.phi

#HT
ht = ak.sum(jets.pt, axis=1)[FCNC_sel]
#MT of lead and subleading lepton with ptmiss (MET)


#get weights of events (scale1fb * generator_weights * lumi)

breakpoint()
if len(FCNC_sel[FCNC_sel]) > 0:
    BDT_param_dict = {"Most_Forward_pt":most_forward_pt,
                      "HT":ht,
                      "LeadLep_eta":np.abs(leadlep_eta),
                      "MET_pt":MET_pt,
                      "LeadLep_pt":leadlep_pt,
                      "LeadLep_dxy":np.abs(leadlep_dxy),
                      "LeadLep_dz":np.abs(leadlep_dz),
                      "SubLeadLep_pt":subleadlep_pt,
                      "SubLeadLep_eta":np.abs(subleadlep_eta),
                      "SubLeadLep_dxy":np.abs(subleadlep_dxy),
                      "SubLeadLep_dz":np.abs(subleadlep_dz),
                      "nJet":njets,
                      "nbtag":nbtag,
                      "LeadJet_pt":leadjet_pt,
                      "SubLeadJet_pt":subleadjet_pt,
                      "SubSubLeadJet_pt":subsubleadjet_pt,
                      "nElectron":nelectron,
                      "MET_pt":MET_pt,
                      "LeadBtag_pt":leadbtag_pt,
                      "MT_LeadLep_MET":mt_leadlep_met,
                      "MT_SubLeadLep_MET":mt_leadlep_met,
                      "LeadLep_SubLeadLep_Mass":leadlep_subleadlep_mass,
                      "weight":1.0
                      }
    #create pandas dataframe
    passed_events = ev[FCNC_sel]
    event_p = ak.to_pandas(passed_events[["event"]])
    for param in BDT_param_dict.keys():
        event_p[param] = BDT_param_dict[param]