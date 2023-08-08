# install.packages('spatialreg')
# install.packages('raster')
# install.packages('modeest')
# install.packages('msos')
library(expm)
library(spatialreg)
library(raster)
library(spdep)
library(terra)
library(stats)
library(coda)
library(modeest)
library(msos)

# beta_coeffs_file = 'C:\\Users\\Jesse Frost\\Documents\\GitHub\\msc-research\\array_beta_coeffs_orig.csv'
# kappa_coeffs_file = 'C:\\Users\\Jesse Frost\\Documents\\GitHub\\msc-research\\array_kappa_coeffs_orig.csv'
# psi_coeffs_file = 'C:\\Users\\Jesse Frost\\Documents\\GitHub\\msc-research\\array_psi_coeffs_orig.csv'
beta_coeffs_file = 'C:\\Users\\Jesse Frost\\Documents\\GitHub\\msc-research\\array_beta_coeffs_nsrp.csv'
kappa_coeffs_file = 'C:\\Users\\Jesse Frost\\Documents\\GitHub\\msc-research\\array_kappa_coeffs_nsrp.csv'
psi_coeffs_file = 'C:\\Users\\Jesse Frost\\Documents\\GitHub\\msc-research\\array_psi_coeffs_nsrp.csv'

beta_df = read.csv(beta_coeffs_file, header=F)
kappa_df = read.csv(kappa_coeffs_file, header=F)
psi_df = read.csv(psi_coeffs_file, header=F)
locations = as.matrix(beta_df[, 1 : 2])
beta_df =  as.matrix(beta_df[, -1 : -2])
kappa_df =  as.matrix(kappa_df[, -1 : -2])
psi_df =  as.matrix(psi_df[, -1 : -2])

# Switch locations matrix column order for distance function
lats = locations[, 1]
lons = locations[, 2]
locations = cbind(lons, lats)
gdis = as.matrix(distance(locations, lonlat=TRUE))
weights = 1 / gdis
weights[!is.finite(weights)] = 0

# Set non-adjacent weights to zero
adjacency = weights * 0
for (i in 1 : dim(locations)[1]) {
  lon_i = locations[i, 1]
  lat_i = locations[i, 2]
  for (j in 1 : dim(locations)[1]) {
    lon_j = locations[j, 1]
    lat_j = locations[j, 2]
    horiz_adjacent = abs(lon_i - lon_j) == 1 && abs(lat_i - lat_j) == 0
    vert_adjacent = abs(lon_i - lon_j) == 0 && abs(lat_i - lat_j) == 0.5
    diag_adjacent = abs(lon_i - lon_j) == 1 && abs(lat_i - lat_j) == 0.5
    # Queen weights; remove diag_adjacent condition for rook weights instead
    if (horiz_adjacent || vert_adjacent || diag_adjacent) {
      adjacency[i, j] = 1
    }
  }
}
weights = weights * adjacency

# Normalise rows
weights = weights / rowSums(weights, na.rm=TRUE)
weights[!is.finite(weights)] = 0

# Test for autocorrelation of the regression coefficients
set.ZeroPolicyOption(TRUE)
weights_listw = mat2listw(weights)
print('beta:')
for (j in 1 : length(colnames(beta_df))) {
  print(c(
    autocor(beta_df[, j], weights, 'moran'),
    as.vector(moran.test(beta_df[, j], weights_listw) $ estimate)[1]
  ))
}
print('kappa:')
for (j in 1 : length(colnames(kappa_df))) {
  print(c(
    autocor(kappa_df[, j], weights, 'moran'),
    as.vector(moran.test(kappa_df[, j], weights_listw) $ estimate)[1]
  ))
}
print('psi:')
for (j in 1 : length(colnames(psi_df))) {
  print(c(
    autocor(psi_df[, j], weights, 'moran'),
    as.vector(moran.test(psi_df[, j], weights_listw) $ estimate)[1]
  ))
}

W = weights
A_func = function(rho, n) {
  A = diag(n) - rho * W
  # Inversion and multiplication are quicker for sparse matrices:
  return(Matrix(A, sparse=TRUE))
}

find_rho_mle = function(y, vicinity_rho_vec=FALSE) {
  n = length(y)
  d_rho_log_L = function(rho) {
    A = A_func(rho, n)
    A_inv = solve(A)
    t1_num = n * t(y) %*% W %*% A %*% y
    t1_den = t(y) %*% t(A) %*% A %*% y
    t2 = sum(diag(-A_inv %*% W))
    return(as.numeric(t1_num / t1_den + t2))
  }
  log_L = function(rho) {
    A = A_func(rho, n)
    t1 = -n/2 * log(t(y) %*% t(A) %*% A %*% y)
    t2 = logdet(as.matrix(A))
    return(as.numeric(t1 + t2))
  }

  rho_vec = seq(-0.5, 1.5, 0.01)
  log_L_vec = c()
  if (isFALSE(vicinity_rho_vec)) {
    # Firstly, produce a rough plot over an extended domain of rho
    # to get an idea about where its maximum lies
    for (i in 1 : length(rho_vec)) {
      log_L_vec[i] = log_L(rho_vec[i])
      # print(c(rho_vec[i], log_L_vec[i]))
    }
    plot(rho_vec, log_L_vec)
    nlm_result = nlm(function(x) {-log_L(x)}, rho_vec[1])
  } else {
    # With a refined vicinity of rho, plot again and solve for the maximum
    for (i in 1 : length(vicinity_rho_vec)) {
      log_L_vec[i] = log_L(vicinity_rho_vec[i])
      print(c(vicinity_rho_vec[i], log_L_vec[i]))
    }
    plot(vicinity_rho_vec, log_L_vec)
    nlm_result = nlm(function(x) {-log_L(x)}, vicinity_rho_vec[1])
  }
  print(nlm_result)
  return(nlm_result $ estimate)
}

find_sigma_mle = function (y, rho) {
  n = length(y)
  Ay = A_func(rho, n) %*% y
  return(as.numeric(1/n * crossprod(Ay)))
}

beta_corrected_df = beta_df * 0
kappa_corrected_df = kappa_df * 0
psi_corrected_df = psi_df * 0
print('beta:')
for (i in 1 : 7) {
  print(i)
  rho = find_rho_mle(beta_df[, i])
  # rho = find_rho_mle(beta_df[, i], seq(0.9, 1.1, 0.01))
  # sigma = find_sigma_mle(beta_df[, i], rho_prec)
  beta_corrected_df[, i] = rho * W %*% beta_df[, i]
}
print('kappa:')
for (i in 1 : 5) {
  print(i)
  rho = find_rho_mle(kappa_df[, i])
  # rho = find_rho_mle(kappa_df[, i], seq(0.9, 1.1, 0.01))
  # sigma = find_sigma_mle(kappa_df[, i], rho_prec)
  kappa_corrected_df[, i] = rho * W %*% kappa_df[, i]
}
print('psi:')
for (i in 1 : 15) {
  print(i)
  rho = find_rho_mle(psi_df[, i])
  # rho = find_rho_mle(psi_df[, i], seq(0.9, 1.1, 0.01))
  # sigma = find_sigma_mle(psi_df[, i], rho_prec)
  psi_corrected_df[, i] = rho * W %*% psi_df[, i]
}

# beta_output_csv = 'C:\\Users\\Jesse Frost\\Documents\\GitHub\\msc-research\\corrected_beta_coeffs_orig.csv'
# kappa_output_csv = 'C:\\Users\\Jesse Frost\\Documents\\GitHub\\msc-research\\corrected_kappa_coeffs_orig.csv'
# psi_output_csv = 'C:\\Users\\Jesse Frost\\Documents\\GitHub\\msc-research\\corrected_psi_coeffs_orig.csv'
beta_output_csv = 'C:\\Users\\Jesse Frost\\Documents\\GitHub\\msc-research\\corrected_beta_coeffs_nsrp.csv'
kappa_output_csv = 'C:\\Users\\Jesse Frost\\Documents\\GitHub\\msc-research\\corrected_kappa_coeffs_nsrp.csv'
psi_output_csv = 'C:\\Users\\Jesse Frost\\Documents\\GitHub\\msc-research\\corrected_psi_coeffs_nsrp.csv'
write.csv(beta_corrected_df, beta_output_csv, row.names=FALSE)
write.csv(kappa_corrected_df, kappa_output_csv, row.names=FALSE)
write.csv(psi_corrected_df, psi_output_csv, row.names=FALSE)