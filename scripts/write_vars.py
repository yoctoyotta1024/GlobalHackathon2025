### Load modules
from IPython.display import display
import intake
import pandas as pd
import sys

from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # don't warn us about future package conflicts

### ------------- INPUT PARAMETERS ------------- ###
current_location = "EU"
zoom = 5
models = [
    "ERA5",
    "JRA3Q",
    "MERRA2",
    "icon_d3hp003",
    "casesm2_10km_nocumulus",
    "nicam_gl11",
    "um_glm_n2560_RAL3p3",
    "ifs_tco3999-ng5_rcbmf",
]
output_dir = Path.cwd() / "bin"
### -------------------------------------------- ###


### Function definitions
def print_var_attributes(ds, output_file):
    with open(output_file, "w") as f:
        sys.stdout = f

        # Dataset overview
        print(f"Dataset dimensions: {dict(ds.sizes)}")
        print(f"Number of variables: {len(ds.data_vars)}")
        # Print vertical levels if they exist
        if "lev" in ds:
            print(f"Vertical levels: {ds['lev'].values}")
        print("-" * 80)

        # Process each variable
        for var_name in ds.data_vars:
            var = ds[var_name]

            # Display variable information with bold name
            display(f"\n## {var_name}")
            print(f"Dimensions: {dict(var.sizes)}")
            print(f"Data type: {var.dtype}")

            # Display attributes in a table if they exist
            if var.attrs:
                print("\nAttributes:")
                pd.set_option("display.max_columns", None)
                pd.set_option(
                    "display.width", 200
                )  # or float('inf') with recent versions
                pd.set_option("display.max_colwidth", None)
                attrs_df = pd.DataFrame(
                    list(var.attrs.items()), columns=["Name", "Value"]
                )
                display(attrs_df)
            else:
                print("\nNo attributes found for this variable")

            print("-" * 80)


### Load catalog
cat = intake.open_catalog(
    "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
)[current_location]
print("--- Datasts in catalog ---\n", "\n".join(list(cat)))

### for each model in 'models' and given zoom level, write out variable metadata to .txt file
print("--- Using models ---\n", "\n".join(models))
for m in models:
    try:
        sys.stdout = sys.__stdout__
        ds = cat[m](zoom=zoom).to_dask()
        output_file = output_dir / f"{m}_zoom{zoom}.txt"
        print(f"writing model {m} at zoom level {zoom} to file {output_file}")
        print_var_attributes(ds, output_file)
    except ValueError:
        sys.stdout = sys.__stdout__
        print(f"failed to load {m} at zoom level {zoom}")
