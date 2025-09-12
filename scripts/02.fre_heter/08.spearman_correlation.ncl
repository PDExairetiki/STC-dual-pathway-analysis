; Plot Spearman correlation distribution of filtered time series and original variables
; Input: filtered data mam_sml14icp_gms_filtered.nc and mam_t2m4icp_gms_filtered.nc
; Output: four spatial correlation distribution maps

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

begin

print("Start calculating Spearman correlation distribution...")

; 1. Read filtered time series data
print("1. Reading filtered time series data...")

; Read SML1 filtering results
f_sml1_filtered = addfile("mam_sml12ts4icp_gms_filtered.nc", "r")
ts_low_SM = f_sml1_filtered->ts_low             ; Low frequency SM
ts_high_SM = f_sml1_filtered->ts_high           ; High frequency SM
ts_low_SM = dtrend(ts_low_SM,False)
ts_high_SM = dtrend(ts_high_SM,False)

; Read T2M filtering results
f_t2m_filtered = addfile("mam_t2m2ts4icp_gms_filtered.nc", "r")
ts_low_T = f_t2m_filtered->ts_low,(/1,2/)      ; Low frequency T
ts_high_T = f_t2m_filtered->ts_high,(/1,2/)    ; High frequency T
ts_low_T = dtrend(ts_low_T,False)
ts_high_T = dtrend(ts_high_T,False)

; 2. Read original spatial data
print("2. Reading original spatial data...")

; Read original SML1 data
f_sml1_orig = addfile("/users/panwei/discipulus/postdoc/0#llp-icp/era/el_sml14icp.nc", "r")
sml1_orig = f_sml1_orig->sml1

; Read original T2M data
f_t2m_orig = addfile("/users/panwei/discipulus/postdoc/0#llp-icp/era/el_t2m4icp.nc", "r")
t2m_orig = f_t2m_orig->t2m

; 3. Data preprocessing
print("3. Data preprocessing...")

; Get dimension information
dims_sml1 = dimsizes(sml1_orig)
dims_t2m = dimsizes(t2m_orig)
ny = dims_sml1(0)  ; Number of years
nt = dims_sml1(1)  ; Number of time steps
nlat = dims_sml1(2)
nlon = dims_sml1(3)

print("Data dimensions: " + ny + " years, " + nt + " time steps, " + nlat + " latitude, " + nlon + " longitude")

; Select spring data (59:150, 92 days)
spring_start = 59
spring_end = 150
spring_days = spring_end - spring_start + 1

; 4. Calculate Spearman correlation coefficients
print("4. Calculating Spearman correlation coefficients...")

; Initialize correlation coefficient arrays
corr_low_SM = new((/nlat, nlon/), float)
corr_high_SM = new((/nlat, nlon/), float)
corr_low_T = new((/nlat, nlon/), float)
corr_high_T = new((/nlat, nlon/), float)

corr_low_SM@_FillValue = -999.0
corr_high_SM@_FillValue = -999.0
corr_low_T@_FillValue = -999.0
corr_high_T@_FillValue = -999.0

; Copy coordinate information
corr_low_SM!0 = "lat"
corr_low_SM!1 = "lon"
corr_low_SM&lat = sml1_orig&lat
corr_low_SM&lon = sml1_orig&lon

corr_high_SM!0 = "lat"
corr_high_SM!1 = "lon"
corr_high_SM&lat = sml1_orig&lat
corr_high_SM&lon = sml1_orig&lon

corr_low_T!0 = "lat"
corr_low_T!1 = "lon"
corr_low_T&lat = t2m_orig&lat
corr_low_T&lon = t2m_orig&lon

corr_high_T!0 = "lat"
corr_high_T!1 = "lon"
corr_high_T&lat = t2m_orig&lat
corr_high_T&lon = t2m_orig&lon

; Define GMS region boundaries
gms_lat_min = 10.0
gms_lat_max = 29.0
gms_lon_min = 92.0
gms_lon_max = 110.0

; Calculate correlation coefficients
do ilat = 0, nlat-1
  do ilon = 0, nlon-1
    
    ; Check if within GMS region
    lat_val = sml1_orig&lat(ilat)
    lon_val = sml1_orig&lon(ilon)
    
    if (lat_val .ge. gms_lat_min .and. lat_val .le. gms_lat_max .and. \
        lon_val .ge. gms_lon_min .and. lon_val .le. gms_lon_max) then
      
      ; Extract spring time series for this grid point
      sml1_spring = sml1_orig(:, spring_start:spring_end, ilat, ilon)
      t2m_spring = t2m_orig(:, spring_start:spring_end, ilat, ilon)
      
      ; Calculate regional average time series
      sml1_ts = dim_avg_n_Wrap(sml1_spring, (/0,1/))
      t2m_ts = dim_avg_n_Wrap(t2m_spring, (/0,1/))
      
      ; Calculate correlation coefficients (using Pearson correlation)
      if (.not. any(ismissing(sml1_ts)) .and. .not. any(ismissing(ts_low_SM))) then
        corr_low_SM(ilat, ilon) = escorc(sml1_ts, ts_low_SM)
        corr_high_SM(ilat, ilon) = escorc(sml1_ts, ts_high_SM)
      end if
      
      if (.not. any(ismissing(t2m_ts)) .and. .not. any(ismissing(ts_low_T))) then
        corr_low_T(ilat, ilon) = escorc(t2m_ts, ts_low_T)
        corr_high_T(ilat, ilon) = escorc(t2m_ts, ts_high_T)
      end if
      
    else
      ; Set missing values for non-GMS regions
      corr_low_SM(ilat, ilon) = corr_low_SM@_FillValue
      corr_high_SM(ilat, ilon) = corr_high_SM@_FillValue
      corr_low_T(ilat, ilon) = corr_low_T@_FillValue
      corr_high_T(ilat, ilon) = corr_high_T@_FillValue
    end if
    
  end do
  
  ; Display progress
  if (mod(ilat, 5) .eq. 0) then
    print("Processing progress: " + ilat + "/" + (nlat-1) + " latitude")
  end if
  
end do

; 5. Save correlation coefficients to file
print("5. Saving correlation coefficients to file...")

system("rm -f spearman_correlation.nc")
f_out = addfile("spearman_correlation.nc", "c")
f_out->corr_low_SM = corr_low_SM
f_out->corr_high_SM = corr_high_SM
f_out->corr_low_T = corr_low_T
f_out->corr_high_T = corr_high_T

; 6. Plot spatial distribution maps
print("6. Plotting spatial distribution maps...")

wks = gsn_open_wks("png", "spearman_correlation_distribution")

; Set up 2×2 panel
plot = new(4, graphic)

; Basic plot resources
res = True
res@gsnDraw = False
res@gsnFrame = False
res@gsnMaximize = True

; Map settings
res@mpProjection = "CylindricalEquidistant"
res@mpCenterLonF = 100.0
res@mpMinLatF = 5.0
res@mpMaxLatF = 30.0
res@mpMinLonF = 90.0
res@mpMaxLonF = 110.0

res@mpFillOn = True
res@mpOceanFillColor = "gray90"
res@mpLandFillColor = "gray90"
res@mpInlandWaterFillColor = "gray90"

res@mpGridAndLimbOn = True
res@mpGridLineColor = "gray50"
res@mpGridLineThicknessF = 0.5

; Contour settings
res@cnFillOn = True
res@cnLinesOn = False
res@cnLineLabelsOn = False
res@cnInfoLabelOn = False

res@cnLevelSelectionMode = "ManualLevels"
res@cnMinLevelValF = -1.0
res@cnMaxLevelValF = 1.0
res@cnLevelSpacingF = 0.2

; Color bar settings
res@lbOrientation = "Horizontal"
res@lbLabelFontHeightF = 0.012
res@lbTitleFontHeightF = 0.014
res@lbTitleString = "Spearman Correlation"

; Title settings
res@gsnLeftStringFontHeightF = 0.02
res@gsnRightStringFontHeightF = 0.02

; Plot (a) SM low frequency correlation
res@gsnLeftString = "(a) SM Low-frequency"
res@gsnRightString = "Correlation"
plot(0) = gsn_csm_contour_map(wks, corr_low_SM, res)

; Plot (b) SM high frequency correlation
res@gsnLeftString = "(b) SM High-frequency"
res@gsnRightString = "Correlation"
plot(1) = gsn_csm_contour_map(wks, corr_high_SM, res)

; Plot (c) T2M low frequency correlation
res@gsnLeftString = "(c) T2M Low-frequency"
res@gsnRightString = "Correlation"
plot(2) = gsn_csm_contour_map(wks, corr_low_T, res)

; Plot (d) T2M high frequency correlation
res@gsnLeftString = "(d) T2M High-frequency"
res@gsnRightString = "Correlation"
plot(3) = gsn_csm_contour_map(wks, corr_high_T, res)

; 7. Combine plots
print("7. Combining plots...")

resP = True
resP@gsnMaximize = True
resP@gsnPanelFigureStrings = (/"", "", "", ""/)
resP@gsnPanelFigureStringsFontHeightF = 0.015
resP@gsnPanelFigureStringsPerimOn = False
resP@gsnPanelFigureStringsBackgroundFillColor = "white"
resP@gsnPanelFigureStringsFontColor = "black"

resP@gsnPanelLeft = 0.05
resP@gsnPanelRight = 0.95
resP@gsnPanelTop = 0.95
resP@gsnPanelBottom = 0.05

resP@gsnPanelYWhiteSpacePercent = 5
resP@gsnPanelXWhiteSpacePercent = 5

gsn_panel(wks, plot, (/2,2/), resP)

print("Spearman correlation distribution maps completed!")
print("Output file: spearman_correlation_distribution.png")
print("Data file: spearman_correlation.nc")

end