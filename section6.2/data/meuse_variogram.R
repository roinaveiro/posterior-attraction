# ------------------------------------------------------------------
# 0.  Packages
# ------------------------------------------------------------------
library(sp);   library(dplyr);   
library(geoR)

# ------------------------------------------------------------------
# 1.  Load Meuse data and reproduce your preprocessing
# ------------------------------------------------------------------
data(meuse, package = "sp")

meuse <- meuse %>%
  mutate(
    log_zinc = log(zinc),          # response
    dist_sq  = sqrt(dist),         # covariate
    ffreq    = factor(ffreq),
    soil     = factor(soil),
    z_elev   = scale(elev)[, 1],
    z_om     = scale(om)[, 1],
    # Shift coords to (0,0) and rescale to kilometres
    x_km     = (x - min(x))/1000,
    y_km     = (y - min(y))/1000
  )

meuse <- na.omit(meuse)
# ------------------------------------------------------------------
# 2.  Fit your linear model (no spatial term)
# ------------------------------------------------------------------
ols_fit <- lm(
  log_zinc ~ dist_sq + z_elev + z_om,  # soil excluded as you chose
  data = meuse
)

# ------------------------------------------------------------------
# 3.  Build a geoR 'geodata' object with the RESIDUALS
#     (variogram on residuals removes large‑scale trend)
# ------------------------------------------------------------------
geo  <- as.geodata(
  cbind(meuse$x_km, meuse$y_km, resid = residuals(ols_fit)),
  coords.col = 1:2,   # x_km, y_km
  data.col   = 3      # residuals
)

# ------------------------------------------------------------------
# 4.  Empirical variogram
#     Choose max.dist as half the maximum interpoint distance
# ------------------------------------------------------------------
emp_vario <- variog(geo, max.dist = 0.5* max(dist(geo$coords)))

plot(emp_vario, pch = 19)        # visual check (optional)

# ------------------------------------------------------------------
# 5.  Fit exponential model with nugget using variofit
#     'ini.cov.pars' = (partial sill, phi) initial guesses
# ------------------------------------------------------------------
ini_pars <- c(var(emp_vario$v),  0.5)   # crude start: variance & 0.5 km
vario_mod <- variofit(
  emp_vario,
  ini.cov.pars = ini_pars,
  nugget = 0.1 * var(emp_vario$v),   # 10% of total var as starting nugget
  cov.model = "exponential",
  weights = "equal"
)

print(vario_mod)

# ------------------------------------------------------------------
# 6.  Extract reasonable hyper‑parameters for PyTorch model
# ------------------------------------------------------------------
sigma2_hat   <- vario_mod$cov.pars[1]   # partial sill  (process variance)
phi_hat_km   <- vario_mod$cov.pars[2]   # decay parameter (km)
nugget_hat   <- vario_mod$nugget        # tau^2

delta2_hat   <- nugget_hat / sigma2_hat # tau^2 / sigma^2

cat("\nSuggested priors for ConjugateSpatialLM:\n",
    "  phi    =", round(phi_hat_km, 3), "per km\n",
    "  delta2 =", round(delta2_hat, 3), "\n")

gamma <- function(h){
  return(sigma2_hat + nugget_hat - sigma2_hat*exp(-h/phi_hat_km))
}

h_seq <- seq(0,2,0.1)
gamma_seq <- NULL
for(i in h_seq){
  print(i)
  gamma_seq <- c(gamma_seq, gamma(i))
}


plot(emp_vario, pch = 19)
lines(h_seq, gamma_seq)

