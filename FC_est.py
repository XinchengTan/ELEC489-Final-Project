# ELEC 489 Final Project
# Author: Xincheng Tan (xt12)
from copy import deepcopy
from itertools import combinations
from inverse_covariance import QuicGraphicalLassoCV, QuicGraphicalLassoEBIC
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import GraphicalLassoCV


# Data Transformations
def anscombe(x):
    '''
    Compute the anscombe variance stabilizing transform.
      the input   x   is noisy Poisson-distributed data
      the output  fx  has variance approximately equal to 1.
    Reference: Anscombe, F. J. (1948), "The transformation of Poisson,
    binomial and negative-binomial data", Biometrika 35 (3-4): 246-254
    '''
    return 2.0*np.sqrt(x + 3.0/8.0)


rng = np.random.RandomState(300)
ss = StandardScaler()
bc = PowerTransformer(method='box-cox')
yj = PowerTransformer(method='yeo-johnson')
qt = QuantileTransformer(n_quantiles=500, output_distribution='normal',
                         random_state=rng)


# Plots the original calcium imaging activity of a neuron
def plot_original(X):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(X, bins=40, density=True)
    ax.set_xlabel('Neuron Activity')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Calcium Imaging Trace')
    return


# Plot transformed data and the fitted gaussian pdf
def plot_transformed(trans_X, method=""):
    X = deepcopy(trans_X)
    X.sort()
    mean, std = stats.norm.fit(X, loc=0)
    pdf_norm = stats.norm.pdf(X, mean, std)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(X, bins=40, density=True)
    ax.plot(X, pdf_norm, label='Fitted normal distribution')
    ax.set_xlabel('Neuron Activity')
    ax.set_ylabel('Transformed Frequency')
    ax.set_title('%s Transformed Distribution of Calcium Imaging Trace' % method)
    ax.legend()


# Use graphical model to estimate functional connectivity
def est_connectivity(X, gm="Glasso", assume_centered=False):
  if gm == "QuicGlasso-CV":
    quic = QuicGraphicalLassoCV(cv=5)
    quic.fit(X)
    return quic.covariance_, quic.precision_, quic.lam_

  elif gm == "QuicGlasso-BIC":
    quic_bic = QuicGraphicalLassoEBIC(gamma=0)
    quic_bic.fit(X)
    return quic_bic.covariance_, quic_bic.precision_, quic_bic.lam_

  else:  # Default: Glasso
    glasso = GraphicalLassoCV(assume_centered=assume_centered, cv=5).fit(X)
    return glasso.covariance_, glasso.get_precision(), glasso.alpha_


def standardize_prec(prec):
  p = len(prec)
  sqrt_diag = np.diag(prec) ** 0.5
  standardizer = np.reshape(sqrt_diag, (p, 1)) * np.reshape(sqrt_diag, (1, p))
  std_prec = prec * standardizer
  return std_prec


def prec2adj_mat(prec, standardize=True, tol=None, qt=0.05, include_negs=False):
  std_prec = standardize_prec(prec) if standardize else prec
  A = np.zeros_like(std_prec)
  P = np.abs(std_prec)
  if tol is None:
    tol = np.quantile(P[P != 0], qt)  # prec is flattened in the computation
  if include_negs:
    A[std_prec > tol] = 1.0
    A[std_prec < -tol] = -1.0
  else:
    A[P > tol] = 1.0
  # print("All prec > tol?", np.all(prec > tol))
  # print("Min prec:", np.min(prec), "Max prec:", np.max(prec))
  return A


def plot_prec(prec, alpha, ax=None, standardize=True, label="", cmap="viridis"):
  P = np.array(prec)
  if standardize:
    P = standardize_prec(prec)
  if ax:
    sns.heatmap(P, cmap=cmap, ax=ax)
  else:
    ax = sns.heatmap(P, cmap=cmap)
  ax.set_xlabel("Neurons")
  ax.set_ylabel("Neurons")
  ax.set_title(r"Precision Matrix [%s, $\lambda$ = %.2f]" % (label, alpha))


def plot_adj_mat(A, ax=None, label="", include_negs=False):
  A2 = A * 3000
  cmap = "cividis" if not include_negs else "bwr"
  if ax:
    sns.heatmap(A2, cmap=cmap, ax=ax)
  else:
    ax = sns.heatmap(A2, cmap=cmap)
  ax.set_xlabel("Neurons")
  ax.set_ylabel("Neurons")
  total_edges = len(A[A != 0])
  if not include_negs:
    ax.set_title("Adjacency Matrix [%s] [%d edges]" % (label, total_edges))
  else:
    pos_edges = len(A[A > 0])
    neg_edges = len(A[A < 0])
    ax.set_title("Adjacency Matrix [%s] [%d edges: %d+, %d-]" % (label, total_edges, pos_edges, neg_edges))


def plot_prec_distri(prec, ax=None, exclude_diag=False, label=""):
  if ax is None:
    fig, ax = plt.subplots(figsize=(8, 5))
  p = np.shape(prec)[0]
  upper_prec = prec[np.triu_indices(p, 1) if exclude_diag else np.triu_indices(p)]

  ax.hist(upper_prec, bins=50)
  ax.set_xlabel("Pairwise Partial Correlation")
  ax.set_ylabel("Frequency")
  ax.set_title("Distribution of pairwise partial correlation [%s]" % label)


# Compare graph structures
def matching_pct(A1, A2):
  # Exclude diagonal entries
  assert np.shape(A1) == np.shape(A2), "Input matrices must have the same shape!"
  p = np.shape(A1)[0]

  upper1, upper2 = A1[np.triu_indices(p, 1)], A2[np.triu_indices(p, 1)]
  diff = np.abs(upper1 - upper2)
  return 1.0 - np.mean(diff)


# Common edge percentage in G1 and G2 with binary adjacency matrix (0, 1)
def overlap_pcts_binary(A1, A2):
  # Exclude diagonal entries
  assert np.shape(A1) == np.shape(A2), "Input matrices must have the same shape!"
  p = np.shape(A1)[0]

  upp1, upp2 = A1[np.triu_indices(p, 1)], A2[np.triu_indices(p, 1)]
  diff = upp1 - upp2
  p1 = 1.0 - len(diff[diff == 1]) / len(upp1[upp1 == 1])
  p2 = 1.0 - len(diff[diff == -1]) / len(upp2[upp2 == 1])
  return p1, p2


# Common edge percentage in G1 and G2 with ternary adjacency matrix (-1, 0, 1)
def overlap_pcts_ternary(A1, A2):
  # Exclude diagonal entries
  assert np.shape(A1) == np.shape(A2), "Input matrices must have the same shape!"
  p = np.shape(A1)[0]

  upp1, upp2 = A1[np.triu_indices(p, 1)], A2[np.triu_indices(p, 1)]
  pos_p1 = len(upp2[(upp1 == 1) & (upp2 == 1)]) / len(upp1[upp1 == 1])
  pos_p2 = len(upp1[(upp1 == 1) & (upp2 == 1)]) / len(upp2[upp2 == 1])
  neg_p1 = len(upp2[(upp1 == -1) & (upp2 == -1)]) / len(upp1[upp1 == -1])
  neg_p2 = len(upp1[(upp1 == -1) & (upp2 == -1)]) / len(upp2[upp2 == -1])
  return pos_p1, pos_p2, neg_p1, neg_p2


def compare_2(A1, A2, lbl1="", lbl2="", include_negs=False):
  match_pct = matching_pct(A1, A2)
  print("Matching percentage between %s and %s: %.2f" % (lbl1, lbl2, match_pct))

  if not include_negs:
    overlap1, overlap2 = overlap_pcts_binary(A1, A2)
    print("Overlap edge percentage wrt %s: %.2f" % (lbl1, overlap1))
    print("Overlap edge percentage wrt %s: %.2f\n" % (lbl2, overlap2))
  else:
    pos_p1, pos_p2, neg_p1, neg_p2 = overlap_pcts_ternary(A1, A2)
    print("Overlap positive edge percentage wrt %s: %.2f" % (lbl1, pos_p1))
    print("Overlap positive edge percentage wrt %s: %.2f\n" % (lbl2, pos_p2))
    print("Overlap negative edge percentage wrt %s: %.2f" % (lbl1, neg_p1))
    print("Overlap negative edge percentage wrt %s: %.2f\n" % (lbl2, neg_p2))


def graph_compare_all(norm_A, anscombe_A, yj_A, qt_A, include_negs=False):
  compare_2(norm_A, anscombe_A, lbl1="Normalization", lbl2="Anscombe transformation", include_negs=include_negs)
  compare_2(norm_A, yj_A, lbl1="Normalization", lbl2="Yeo-Johnson transformation", include_negs=include_negs)
  compare_2(norm_A, qt_A, lbl1="Normalization", lbl2="Quantile transformation", include_negs=include_negs)
  compare_2(anscombe_A, yj_A, lbl1="Anscombe transformation", lbl2="Yeo-Johnson transformation", include_negs=include_negs)
  compare_2(anscombe_A, qt_A, lbl1="Anscombe transformation", lbl2="Quantile transformation", include_negs=include_negs)
  compare_2(yj_A, qt_A, lbl1="Yeo-Johnson transformation", lbl2="Quantile transformation", include_negs=include_negs)


# Apply all 4 transformations
def transform_all(x):
    # Normalizaton
    norm_x = ss.fit_transform(x)

    # Anscombe transformation
    anscombe_x = anscombe(x)

    # Yeo-Johnson transformation
    yj_x = yj.fit_transform(x)

    # Quantile transformation
    qt_x = qt.fit_transform(x)

    return norm_x, anscombe_x, yj_x, qt_x


# For each transformation, compare the graph estimated by different graphical models
def compare_diff_gms(gm2outs):
  # gm2outs is a mapping from GM method name to 4 output adjacency matrices
  gm_outs_items = list(gm2outs.items())

  def compare2gms(gm1, gm_outs1, gm2, gm_outs2):
    norm = "Normalization"
    anscombe = "Anscombe"
    yj = "Yeo-Johnson"
    qt = "Quantile"
    compare_2(gm_outs1[0], gm_outs2[0], lbl1="(%s, %s)" % (gm1, norm), lbl2="(%s, %s)" % (gm2, norm))
    compare_2(gm_outs1[1], gm_outs2[1], lbl1="(%s, %s)" % (gm1, anscombe), lbl2="(%s, %s)" % (gm2, anscombe))
    compare_2(gm_outs1[2], gm_outs2[2], lbl1="(%s, %s)" % (gm1, yj), lbl2="(%s, %s)" % (gm2, yj))
    compare_2(gm_outs1[3], gm_outs2[3], lbl1="(%s, %s)" % (gm1, qt), lbl2="(%s, %s)" % (gm2, qt))

  combo_idxs = combinations(range(len(gm2outs)), 2)
  for i1, i2 in combo_idxs:
    compare2gms(gm_outs_items[i1][0], gm_outs_items[i1][1], gm_outs_items[i2][0], gm_outs_items[i2][1])



# Compare the estimated precision matrices among different transformations
def compare_diff_trans(norm_prec, anscombe_prec, yj_prec, qt_prec, norm_alpha, anscombe_alpha, yj_alpha, qt_alpha,
                       dataset="", gm="Glasso", include_negs=False, std_prec=True):
  no_trans = "Normalization"
  anscombe_trans = "Anscombe Transformation"
  yj_trans = "Yeo-Johnson Transformation"
  qt_trans = "Quantile Transformation"

  # Plot precision & adjacency matrices
  fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(15, 26))
  fig.suptitle("Dataset: %s\nModel: %s" % (dataset, gm), fontsize=15)

  norm_A = prec2adj_mat(norm_prec, standardize=std_prec, include_negs=include_negs)
  plot_prec(norm_prec, standardize=std_prec, alpha=norm_alpha, ax=ax1, label=no_trans)
  plot_adj_mat(norm_A, ax=ax2, label=no_trans, include_negs=include_negs)

  anscombe_A = prec2adj_mat(anscombe_prec, standardize=std_prec, include_negs=include_negs)
  plot_prec(anscombe_prec, standardize=std_prec, alpha=anscombe_alpha, ax=ax3, label=anscombe_trans)
  plot_adj_mat(anscombe_A, ax=ax4, label=anscombe_trans, include_negs=include_negs)

  yj_A = prec2adj_mat(yj_prec, standardize=std_prec, include_negs=include_negs)
  plot_prec(yj_prec, ax=ax5, standardize=std_prec, alpha=yj_alpha, label=yj_trans)
  plot_adj_mat(yj_A, ax=ax6, label=yj_trans, include_negs=include_negs)

  qt_A = prec2adj_mat(qt_prec, standardize=std_prec, include_negs=include_negs)
  plot_prec(qt_prec, ax=ax7, standardize=std_prec, alpha=qt_alpha, label=qt_trans)
  plot_adj_mat(qt_A, ax=ax8, label=qt_trans, include_negs=include_negs)

  plt.subplots_adjust(hspace=0.2)
  plt.show()

  # Plot distribution of pairwise partial correlation
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
  fig.suptitle("Dataset: %s\nModel: %s" % (dataset, gm), fontsize=15)
  plot_prec_distri(norm_prec, ax1, label=no_trans)
  plot_prec_distri(anscombe_prec, ax2, label=anscombe_trans)
  plot_prec_distri(yj_prec, ax3, label=yj_trans)
  plot_prec_distri(qt_prec, ax4, label=qt_trans)
  plt.subplots_adjust(hspace=0.2)
  plt.show()

  # Compare edge overlaps
  graph_compare_all(norm_A, anscombe_A, yj_A, qt_A, include_negs)
  return norm_A, anscombe_A, yj_A, qt_A


# Execute the entire transformation - estimation pipeline
def run_all(x, gm="Glasso", dataset="", include_negs=False, std_prec=True):
  # Apply all 4 transformations
  norm_x, anscombe_x, yj_x, qt_x = transform_all(x)

  # Apply graphical model
  norm_cov, norm_prec, norm_alpha = est_connectivity(norm_x, gm=gm)

  anscombe_cov, anscombe_prec, anscombe_alpha = est_connectivity(anscombe_x, gm=gm)

  yj_cov, yj_prec, yj_alpha = est_connectivity(yj_x, gm=gm)

  qt_cov, qt_prec, qt_alpha = est_connectivity(qt_x, gm=gm)

  # Compare the graphical model results
  norm_A, anscombe_A, yj_A, qt_A = compare_diff_trans(norm_prec, anscombe_prec, yj_prec, qt_prec,
                                                      norm_alpha, anscombe_alpha, yj_alpha, qt_alpha,
                                                      dataset, gm=gm, include_negs=include_negs, std_prec=std_prec)
  return norm_A, anscombe_A, yj_A, qt_A

