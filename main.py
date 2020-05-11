from fc_estimation.FC_est import *
import scipy.io as sio
import argparse

if __name__ == "__main__":

  START_TPT = 100
  END_TPT = 10000
  TPT_STEP = 10
  TPTS = range(START_TPT, END_TPT, TPT_STEP)
  NID_MIN = 0
  NID_MAX = 100

  # fsp shape: (neurons, timepoints)
  # Spontaneous Activity
  spont_fp = "../data/stringer_spont_170818.mat"
  fsp_spont = sio.loadmat(spont_fp).get("Fsp")
  print(fsp_spont.shape)
  spont_x = fsp_spont[NID_MIN:NID_MAX, TPTS].T
  run_all(spont_x, gm="Glasso", dataset="Stringer's Neuron Activity Recording (Spontaneous)")

  # Natural Image Stimuli
  stimspont_fp = "../data/stringer_stimspont_170825.mat"
  fsp_stim = sio.loadmat(stimspont_fp).get("Fsp")
  print(fsp_stim.shape)
  stim_x = fsp_stim[:100, TPTS].T
  stim_outs_glasso = run_all(stim_x, gm="Glasso", dataset="Stringer's Neuron Activity Recording (with 32 Images Stimuli)")

  # Drifting Grating
  ori_fp = "../data/stringer_ori32_170817.mat"
  fsp_ori = sio.loadmat(ori_fp).get("Fsp")
  print(fsp_ori.shape)
  ori_x = fsp_ori[:100, TPTS].T
  ori_outs_glasso = run_all(ori_x, gm="Glasso", dataset="Stringer's Neuron Activity Recording (with Drifting Grating)")


