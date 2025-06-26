library(sp)
library(gstat)
library(tidyverse)

# 2. Load the Meuse dataset
data("meuse")  # This loads a data frame called 'meuse' in your global environment.

meuse <- meuse %>% na.omit()

write_csv(meuse, "meuse_raw.csv")
# The 'meuse' data frame has columns:
#   x, y, cadmium, copper, lead, zinc, elev, dist, om, ffreq, soil, lime, landuse, dist.m

# Let's inspect the first few rows:
head(meuse)

# 3. Keep only the columns we need:
#    - x, y: the spatial coordinates (in meters, Dutch RD New)
#    - cadmium, copper, lead, zinc: metal concentrations (ppm)
df_meuse <- meuse[, c("x", "y", "cadmium", "copper", "lead", "zinc", "dist")]

# 4. Remove any rows with NA values, if any (in meuse, typically there's none, but just in case)
df_meuse <- na.omit(df_meuse)

# 5. Inspect the structure
str(df_meuse)

# 6. (Optional) Export to CSV if we want to import from Python
#    Suppose we write it to a local CSV in your working directory:
write.csv(df_meuse, file = "meuse_data.csv", row.names = FALSE)

# Regress lead on cadmium, copper, and zinc
# 7. Fit a linear model
model <- lm(lead ~ cadmium + copper + zinc, data = meuse)
summary(model)
