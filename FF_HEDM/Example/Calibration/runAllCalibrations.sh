# Varex with bad distortion
python ~/opt/MIDAS/utils/AutoCalibrateZarr.py --data CeO2_10s_1000mm_42keV_000718.tiff --n-iterations 5 --px 150.0 --wavelength 0.29519 --material ceo2 --im-trans 1 --cpus 8 --fit-p-models all

# Varex with low distortion
python ~/opt/MIDAS/utils/AutoCalibrateZarr.py --data Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif --n-iterations 5 --px 150.0 --wavelength 0.196793 --material ceo2 --im-trans 2 --cpus 8 --fit-p-models all

# Pilatus with panel shifts and auto masking
# Delete older panel shifts file and mask file to initialize from scratch
rm *panel* *autocal_mask*
python ~/opt/MIDAS/utils/AutoCalibrateZarr.py --data CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tif --dark dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif --px 172.0 --wavelength 0.172979 --material ceo2 --im-trans 2 --n-iterations 5 --cpus 8 --fit-p-models all

# offset GE
python ~/opt/MIDAS/utils/AutoCalibrateZarr.py --data CeO2_1s_65pt351keV_1860mm_000007.edf.ge1 --dark dark_6s_000010.ge1 --px 200.0 --wavelength 0.189714 --material ceo2 --n-iterations 5 --cpus 8 --fit-p-models all


# cleanup at the end
rm *..* *ps.txt refined* autocal.log calibrant_screen_out.csv ci_profiles.csv hkls.csv integrator_*.csv *panel* *autocal_mask*