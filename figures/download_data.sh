#!/usr/bin/env bash
# Fetch the public data products used by the figure scripts.
# Total download ~40 MB, dominated by the Pantheon+ covariance.
set -e
mkdir -p data

P="https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR"
curl -L -o data/PantheonPlusSH0ES.dat         "$P/Pantheon%2BSH0ES.dat"
curl -L -o data/PantheonPlusSH0ES_STATSYS.cov "$P/Pantheon%2BSH0ES_STAT%2BSYS.cov"

B="https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2"
curl -L -o data/desi_dr2_mean.txt "$B/desi_gaussian_bao_ALL_GCcomb_mean.txt"
curl -L -o data/desi_dr2_cov.txt  "$B/desi_gaussian_bao_ALL_GCcomb_cov.txt"

U="https://raw.githubusercontent.com/rubind/union3_release/main"
curl -L -o data/union3_mu.fits  "$U/mu_mat_union3_cosmo%3D2_mu.fits"
curl -L -o data/union31_mu.fits "$U/mu_mat_union3.1_UNITY1.8_template_cosmo%3D2_0_mu.fits"

D="https://raw.githubusercontent.com/des-science/DES-SN5YR/main/4_DISTANCES_COVMAT"
curl -L -o data/DES-Dovekie_HD.csv "$D/DES-Dovekie_HD.csv"
curl -L -o data/DES_STATSYS.npz    "$D/STAT%2BSYS.npz"

echo "done -> data/"
