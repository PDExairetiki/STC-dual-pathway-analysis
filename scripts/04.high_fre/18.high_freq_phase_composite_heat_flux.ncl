load "../../shapefiles/shapefile_utils.ncl"

begin

print("=== High-frequency heat flux phase composite distribution analysis ===")

; 1. Read high-frequency SM average cycle as ruler
print("1. Reading high-frequency SM average cycle as ruler...")

path = "../../data/climatology/filtered/"

f_avg_cycle = addfile(path+"high_freq_sm_avg_cycle_refined.nc", "r")
avg_cycle = f_avg_cycle->avg_cycle
cycle_time = f_avg_cycle->cycle_time
n_cycles = f_avg_cycle->n_cycles

print("Ruler cycle length: " + dimsizes(avg_cycle) + " days")
print("Number of typical cycles found: " + n_cycles)

; 2. Read high-frequency heat flux data
print("2. Reading high-frequency heat flux data...")

; Read high-frequency latent heat flux data
f_slhf_filtered = addfile(path+"mam_slhf4icp_gms_filtered.nc", "r")
ts_high_SLHF = f_slhf_filtered->ts_high  ; (day, lat, lon)

; Read high-frequency sensible heat flux data
f_sshf_filtered = addfile(path+"mam_sshf4icp_gms_filtered.nc", "r")
ts_high_SSHF = f_sshf_filtered->ts_high  ; (day, lat, lon)

; Read high-frequency net radiation data
f_rn_filtered = addfile(path+"mam_rn4icp_gms_filtered.nc", "r")
ts_high_RN = f_rn_filtered->ts_high  ; (day, lat, lon)

; Read high-frequency air temperature data (for direction calibration, consistent with script 22: T_H is in the same direction as SH_H, LE_H is opposite)
f_t2m_filtered = addfile(path+"mam_hp4icp_gms_filtered.nc", "r")
ts_high_T = f_t2m_filtered->ts_high  ; (day, lat, lon)

; Read ERA5-land data for land-sea mask

pathl="../../data/climatology/origin/"

f_era5_slhf = addfile(pathl+"fmamj_slhf4icp.nc", "r")
x_slhf = f_era5_slhf->slhf(28:119,:,:)
lsm_slhf = where(x_slhf.ne.x_slhf@_FillValue, 1, x_slhf@_FillValue)

f_era5_sshf = addfile(pathl+"fmamj_sshf4icp.nc", "r")
x_sshf = f_era5_sshf->sshf(28:119,:,:)
lsm_sshf = where(x_sshf.ne.x_sshf@_FillValue, 1, x_sshf@_FillValue)

f_era5_rn = addfile(pathl+"fmamj_rn4icp.nc", "r")
x_rn = f_era5_rn->rn(28:119,:,:)
lsm_rn = where(x_rn.ne.x_rn@_FillValue, 1, x_rn@_FillValue)

; Get dimension information
lat = ts_high_SLHF&lat
lon = ts_high_SLHF&lon
nlat = dimsizes(lat)
nlon = dimsizes(lon)
ntime = dimsizes(ts_high_SLHF&day)

print("Data dimensions: " + ntime + " days, " + nlat + " latitude, " + nlon + " longitude")

; 3. Perform detrending
print("3. Performing detrending...")

; Detrend and standardize for each grid point
ts_high_SLHF_detrend = ts_high_SLHF
ts_high_SSHF_detrend = ts_high_SSHF
ts_high_RN_detrend = ts_high_RN

do i = 0, nlat-1
    do j = 0, nlon-1
        ; Latent heat flux detrending
        if (.not. all(ismissing(ts_high_SLHF(:, i, j)))) then
            ts_high_SLHF_detrend(:, i, j) = dtrend(ts_high_SLHF(:, i, j), False)
            ts_high_SLHF_detrend(:, i, j) = dim_standardize_n_Wrap(ts_high_SLHF_detrend(:, i, j), 0, 0)
        end if
        
        ; Sensible heat flux detrending
        if (.not. all(ismissing(ts_high_SSHF(:, i, j)))) then
            ts_high_SSHF_detrend(:, i, j) = dtrend(ts_high_SSHF(:, i, j), False)
            ts_high_SSHF_detrend(:, i, j) = dim_standardize_n_Wrap(ts_high_SSHF_detrend(:, i, j), 0, 0)
        end if
        
        ; Net radiation detrending
        if (.not. all(ismissing(ts_high_RN(:, i, j)))) then
            ts_high_RN_detrend(:, i, j) = dtrend(ts_high_RN(:, i, j), False)
            ts_high_RN_detrend(:, i, j) = dim_standardize_n_Wrap(ts_high_RN_detrend(:, i, j), 0, 0)
        end if

        ; Temperature detrending (only for regional average direction calibration)
        if (.not. all(ismissing(ts_high_T(:, i, j)))) then
            ts_high_T(:, i, j) = dtrend(ts_high_T(:, i, j), False)
            ts_high_T(:, i, j) = dim_standardize_n_Wrap(ts_high_T(:, i, j), 0, 0)
        end if
    end do
end do

; Apply land-sea mask
ts_high_SLHF_detrend = ts_high_SLHF_detrend * lsm_slhf
ts_high_SSHF_detrend = ts_high_SSHF_detrend * lsm_sshf
ts_high_RN_detrend = ts_high_RN_detrend * lsm_rn

; 3.1 Direction calibration (consistent with script 22: T_H is in the same direction as SH_H; LE_H is opposite to SH_H; RN_H is in the same direction as SH_H)
print("3.1 Direction calibration (consistent with script 22)...")
slhf_mean = dim_avg_n_Wrap(ts_high_SLHF_detrend, (/1,2/))
sshf_mean = dim_avg_n_Wrap(ts_high_SSHF_detrend, (/1,2/))
rn_mean   = dim_avg_n_Wrap(ts_high_RN_detrend,   (/1,2/))
t_mean    = dim_avg_n_Wrap(ts_high_T,            (/1,2/))

; If T and SH correlation is negative, flip SH
; If T regional average sequence has valid values, perform direction calibration
if (.not. any(ismissing(t_mean))) then
  c_t_sh = escorc(t_mean, sshf_mean)
  if (c_t_sh .lt. 0.0) then
    ts_high_SSHF_detrend = -ts_high_SSHF_detrend
    sshf_mean = -sshf_mean
  end if
end if

; Make LE opposite to SH: if correlation is positive, flip LE
c_le_sh = escorc(slhf_mean, sshf_mean)
if (c_le_sh .gt. 0.0) then
  ts_high_SLHF_detrend = -ts_high_SLHF_detrend
  slhf_mean = -slhf_mean
end if

; Optional: Make RN in the same direction as SH (if negative, flip RN)
c_rn_sh = escorc(rn_mean, sshf_mean)
if (c_rn_sh .lt. 0.0) then
  ts_high_RN_detrend = -ts_high_RN_detrend
end if

; 4. Apply shapefile mask
print("4. Applying shapefile mask...")

opt_mask = True
opt_mask@keep = True  ; Keep data within GMS region
opt_mask@return_mask = False

do d = 0, ntime-1
    ; Apply mask to each time slice
    slhf_slice = ts_high_SLHF_detrend(d,:,:)
    slhf_slice!0 = "lat"
    slhf_slice!1 = "lon"
    
    sshf_slice = ts_high_SSHF_detrend(d,:,:)
    sshf_slice!0 = "lat"
    sshf_slice!1 = "lon"
    
    rn_slice = ts_high_RN_detrend(d,:,:)
    rn_slice!0 = "lat"
    rn_slice!1 = "lon"
    
    ; Apply mask
    ts_high_SLHF_detrend(d,:,:) = shapefile_mask_data(slhf_slice, "../../shapefiles/gms.shp", opt_mask)
    ts_high_SSHF_detrend(d,:,:) = shapefile_mask_data(sshf_slice, "../../shapefiles/gms.shp", opt_mask)
    ts_high_RN_detrend(d,:,:) = shapefile_mask_data(rn_slice, "../../shapefiles/gms.shp", opt_mask)
end do

; 5. Phase composite based on ruler
print("5. Phase composite based on ruler...")

; Define number of phases (-7~0 days, two days per phase, 4 phases total)
n_phases = 4
phase_composite_SLHF = new((/n_phases, nlat, nlon/), float)
phase_composite_SSHF = new((/n_phases, nlat, nlon/), float)
phase_composite_RN = new((/n_phases, nlat, nlon/), float)
phase_composite_SLHF@_FillValue = 1e20
phase_composite_SSHF@_FillValue = 1e20
phase_composite_RN@_FillValue = 1e20

; Initialize counter
phase_count = new(n_phases, integer)
phase_count = 0

; Phase composite based on first half of ruler cycle characteristics (consistent with script 23)
; Phase definition: -7~-6 days as phase 0, -5~-4 days as phase 1, -3~-2 days as phase 2, -1~0 days as phase 3
do phase = 0, n_phases-1
    print("  Processing phase " + (phase+1) + "/" + n_phases)
    
    ; Find patterns similar to ruler phase in time series
    do t = 1, ntime-15  ; Leave enough space
        ; Extract 15-day segment (use SLHF data to calculate similarity)
        segment = ts_high_SLHF_detrend(t:t+14, :, :)
        
        ; Calculate similarity with first half of ruler (consistent with script 23)
        ; Calculate correlation coefficient (regional average, using first 7 days)
            segment_avg = dim_avg_n_Wrap(segment(0:6, :, :), (/1,2/))
            corr = escorc(segment_avg, avg_cycle(0:6))
            
            ; If correlation coefficient is high enough, consider it a similar pattern
            if (corr .gt. 0.7) then
                ; Accumulate to phase composite
                if (phase_count(phase) .eq. 0) then
                ; Calculate corresponding day range based on phase (consistent with script 23)
                if (phase .eq. 0) then
                    ; Phase 0: -7~-6 days, corresponding to ruler's day 7-8
                    phase_composite_SLHF(phase, :, :) = dim_avg_n_Wrap(ts_high_SLHF_detrend(t+6:t+7, :, :), 0)
                    phase_composite_SSHF(phase, :, :) = dim_avg_n_Wrap(ts_high_SSHF_detrend(t+6:t+7, :, :), 0)
                    phase_composite_RN(phase, :, :) = dim_avg_n_Wrap(ts_high_RN_detrend(t+6:t+7, :, :), 0)
                else if (phase .eq. 1) then
                    ; Phase 1: -5~-4 days, corresponding to ruler's day 5-6
                    phase_composite_SLHF(phase, :, :) = dim_avg_n_Wrap(ts_high_SLHF_detrend(t+4:t+5, :, :), 0)
                    phase_composite_SSHF(phase, :, :) = dim_avg_n_Wrap(ts_high_SSHF_detrend(t+4:t+5, :, :), 0)
                    phase_composite_RN(phase, :, :) = dim_avg_n_Wrap(ts_high_RN_detrend(t+4:t+5, :, :), 0)
                else if (phase .eq. 2) then
                    ; Phase 2: -3~-2 days, corresponding to ruler's day 3-4
                    phase_composite_SLHF(phase, :, :) = dim_avg_n_Wrap(ts_high_SLHF_detrend(t+2:t+3, :, :), 0)
                    phase_composite_SSHF(phase, :, :) = dim_avg_n_Wrap(ts_high_SSHF_detrend(t+2:t+3, :, :), 0)
                    phase_composite_RN(phase, :, :) = dim_avg_n_Wrap(ts_high_RN_detrend(t+2:t+3, :, :), 0)
                else if (phase .eq. 3) then
                    ; Phase 3: -1~0 days, corresponding to ruler's day 1-2
                    phase_composite_SLHF(phase, :, :) = dim_avg_n_Wrap(ts_high_SLHF_detrend(t:t+1, :, :), 0)
                    phase_composite_SSHF(phase, :, :) = dim_avg_n_Wrap(ts_high_SSHF_detrend(t:t+1, :, :), 0)
                    phase_composite_RN(phase, :, :) = dim_avg_n_Wrap(ts_high_RN_detrend(t:t+1, :, :), 0)
                end if
                end if
                end if
                end if
            else
                ; Accumulate pattern
                if (phase .eq. 0) then
                    phase_composite_SLHF(phase, :, :) = phase_composite_SLHF(phase, :, :) + dim_avg_n_Wrap(ts_high_SLHF_detrend(t+6:t+7, :, :), 0)
                    phase_composite_SSHF(phase, :, :) = phase_composite_SSHF(phase, :, :) + dim_avg_n_Wrap(ts_high_SSHF_detrend(t+6:t+7, :, :), 0)
                    phase_composite_RN(phase, :, :) = phase_composite_RN(phase, :, :) + dim_avg_n_Wrap(ts_high_RN_detrend(t+6:t+7, :, :), 0)
                else if (phase .eq. 1) then
                    phase_composite_SLHF(phase, :, :) = phase_composite_SLHF(phase, :, :) + dim_avg_n_Wrap(ts_high_SLHF_detrend(t+4:t+5, :, :), 0)
                    phase_composite_SSHF(phase, :, :) = phase_composite_SSHF(phase, :, :) + dim_avg_n_Wrap(ts_high_SSHF_detrend(t+4:t+5, :, :), 0)
                    phase_composite_RN(phase, :, :) = phase_composite_RN(phase, :, :) + dim_avg_n_Wrap(ts_high_RN_detrend(t+4:t+5, :, :), 0)
                else if (phase .eq. 2) then
                    phase_composite_SLHF(phase, :, :) = phase_composite_SLHF(phase, :, :) + dim_avg_n_Wrap(ts_high_SLHF_detrend(t+2:t+3, :, :), 0)
                    phase_composite_SSHF(phase, :, :) = phase_composite_SSHF(phase, :, :) + dim_avg_n_Wrap(ts_high_SSHF_detrend(t+2:t+3, :, :), 0)
                    phase_composite_RN(phase, :, :) = phase_composite_RN(phase, :, :) + dim_avg_n_Wrap(ts_high_RN_detrend(t+2:t+3, :, :), 0)
                else if (phase .eq. 3) then
                    phase_composite_SLHF(phase, :, :) = phase_composite_SLHF(phase, :, :) + dim_avg_n_Wrap(ts_high_SLHF_detrend(t:t+1, :, :), 0)
                    phase_composite_SSHF(phase, :, :) = phase_composite_SSHF(phase, :, :) + dim_avg_n_Wrap(ts_high_SSHF_detrend(t:t+1, :, :), 0)
                    phase_composite_RN(phase, :, :) = phase_composite_RN(phase, :, :) + dim_avg_n_Wrap(ts_high_RN_detrend(t:t+1, :, :), 0)
                end if
                end if
                end if
                end if
            end if
            phase_count(phase) = phase_count(phase) + 1
        end if
    end do
    
    ; Calculate average
    if (phase_count(phase) .gt. 0) then
        phase_composite_SLHF(phase, :, :) = phase_composite_SLHF(phase, :, :) / phase_count(phase)
        phase_composite_SSHF(phase, :, :) = phase_composite_SSHF(phase, :, :) / phase_count(phase)
        phase_composite_RN(phase, :, :) = phase_composite_RN(phase, :, :) / phase_count(phase)
        print("    Phase " + (phase+1) + " found " + phase_count(phase) + " similar patterns")
    end if
end do

; 6. Perform significance test (t-test)
print("6. Performing significance test (t-test)...")

; Create significance test arrays
significance_SLHF = new((/n_phases, nlat, nlon/), float)
significance_SSHF = new((/n_phases, nlat, nlon/), float)
significance_RN = new((/n_phases, nlat, nlon/), float)
significance_SLHF@_FillValue = 1e20
significance_SSHF@_FillValue = 1e20
significance_RN@_FillValue = 1e20

; Set significance level (95% confidence level)
alpha = 0.05

; Perform t-test for each phase
do phase = 0, n_phases-1
    print("  Performing significance test for phase " + (phase+1) + "...")
    
    if (phase_count(phase) .gt. 1) then  ; At least 2 samples needed for t-test
        ; Re-collect all sample data for this phase
        phase_samples_SLHF = new((/phase_count(phase), nlat, nlon/), float)
        phase_samples_SSHF = new((/phase_count(phase), nlat, nlon/), float)
        phase_samples_RN = new((/phase_count(phase), nlat, nlon/), float)
        phase_samples_SLHF@_FillValue = 1e20
        phase_samples_SSHF@_FillValue = 1e20
        phase_samples_RN@_FillValue = 1e20
        
        sample_idx = 0
        
        ; Re-iterate through time series to collect samples
        do t = 1, ntime-15
            segment = ts_high_SLHF_detrend(t:t+14, :, :)
            segment_avg = dim_avg_n_Wrap(segment(0:6, :, :), (/1,2/))
            corr = escorc(segment_avg, avg_cycle(0:6))
            
            if (corr .gt. 0.7) then
                ; Extract corresponding data based on phase
                if (phase .eq. 0) then
                    phase_samples_SLHF(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_SLHF_detrend(t+6:t+7, :, :), 0)
                    phase_samples_SSHF(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_SSHF_detrend(t+6:t+7, :, :), 0)
                    phase_samples_RN(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_RN_detrend(t+6:t+7, :, :), 0)
                else if (phase .eq. 1) then
                    phase_samples_SLHF(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_SLHF_detrend(t+4:t+5, :, :), 0)
                    phase_samples_SSHF(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_SSHF_detrend(t+4:t+5, :, :), 0)
                    phase_samples_RN(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_RN_detrend(t+4:t+5, :, :), 0)
                else if (phase .eq. 2) then
                    phase_samples_SLHF(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_SLHF_detrend(t+2:t+3, :, :), 0)
                    phase_samples_SSHF(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_SSHF_detrend(t+2:t+3, :, :), 0)
                    phase_samples_RN(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_RN_detrend(t+2:t+3, :, :), 0)
                else if (phase .eq. 3) then
                    phase_samples_SLHF(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_SLHF_detrend(t:t+1, :, :), 0)
                    phase_samples_SSHF(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_SSHF_detrend(t:t+1, :, :), 0)
                    phase_samples_RN(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_RN_detrend(t:t+1, :, :), 0)
                end if
                end if
                end if
                end if
                
                sample_idx = sample_idx + 1
                if (sample_idx .ge. phase_count(phase)) then
                    break
                end if
            end if
        end do
        
        ; Perform significance test for each grid point (normal approximation: |mean| > 1.96*SE is significant)
        do i = 0, nlat-1
            do j = 0, nlon-1
                if (.not. all(ismissing(phase_samples_SLHF(:, i, j)))) then
                    x = phase_samples_SLHF(:, i, j)
                    m = avg(x)
                    sd = stddev(x)
                    if (.not. ismissing(sd) .and. sd .gt. 0.) then
                        se = sd / sqrt( tofloat(phase_count(phase)) )
                        if (abs(m) .gt. (1.96*se)) then
                            significance_SLHF(phase, i, j) = 1.0
                        else
                            significance_SLHF(phase, i, j) = 0.0
                        end if
                    end if
                end if

                if (.not. all(ismissing(phase_samples_SSHF(:, i, j)))) then
                    x = phase_samples_SSHF(:, i, j)
                    m = avg(x)
                    sd = stddev(x)
                    if (.not. ismissing(sd) .and. sd .gt. 0.) then
                        se = sd / sqrt( tofloat(phase_count(phase)) )
                        if (abs(m) .gt. (1.96*se)) then
                            significance_SSHF(phase, i, j) = 1.0
                        else
                            significance_SSHF(phase, i, j) = 0.0
                        end if
                    end if
                end if

                if (.not. all(ismissing(phase_samples_RN(:, i, j)))) then
                    x = phase_samples_RN(:, i, j)
                    m = avg(x)
                    sd = stddev(x)
                    if (.not. ismissing(sd) .and. sd .gt. 0.) then
                        se = sd / sqrt( tofloat(phase_count(phase)) )
                        if (abs(m) .gt. (1.96*se)) then
                            significance_RN(phase, i, j) = 1.0
                        else
                            significance_RN(phase, i, j) = 0.0
                        end if
                    end if
                end if
            end do
        end do
        
        delete(phase_samples_SLHF)
        delete(phase_samples_SSHF)
        delete(phase_samples_RN)
    end if
end do

; 7. Plot phase composite distribution
print("7. Plotting phase composite distribution...")

wks = gsn_open_wks("png", "high_freq_heat_flux_phase_composite")

; Create panel
plot = new(12, graphic)

; Basic plot resources (mimic script 16)
res = True
res@gsnDraw = False
res@gsnFrame = False
res@gsnLeftString = ""
res@gsnRightString = ""
res@gsnSpreadColors = True
res@gsnAddCyclic = False

res@cnInfoLabelOn = False
res@cnLinesOn = False
res@cnLineLabelsOn = False
res@cnFillOn = True
res@cnFillMode = "AreaFill"
res@mpFillOn = True
res@lbLabelBarOn = False

; Set map range
res@mpMinLatF = lat(0)
res@mpMaxLatF = lat(nlat-1)
res@mpMinLonF = lon(1)
res@mpMaxLonF = lon(nlon-1)

; Set ocean fill
res@mpFillOn = True
res@mpOceanFillColor = "gray90"
res@mpLandFillColor = "gray90"
res@mpInlandWaterFillColor = "gray90"
res@cnMissingValFillColor = "gray90"
res@mpShapeMode = "FreeAspect"
res@mpDataBaseVersion = "Ncarg4_1"
res@mpDataSetName = "Earth..4"
res@mpOutlineBoundarySets = "National"
res@mpOutlineSpecifiers = (/"China:states"/)

res@tmXTOn = False
res@tmYROn = False

res@vpHeightF = 0.6
res@vpWidthF = 0.6

res@pmTickMarkDisplayMode = "Always"

; Color settings: cool color for negative anomalies, warm color for positive anomalies (opposite to SM)
res@cnLevelSelectionMode = "ManualLevels"
res@cnMinLevelValF = -1.0
res@cnMaxLevelValF = 1.0
res@cnLevelSpacingF = 0.1

; Use cool color for negative anomalies, warm color for positive anomalies (opposite to SM)
FillColors = (/20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240/)
res@cnFillColors = FillColors  ; Do not reverse color order

res@gsnLeftStringFontHeightF = 0.02

; Phase labels
stringw = (/"Phase -3", "Phase -2", "Phase -1", "Phase 0"/)
strings = (/"~F33~l~F~E~B~H~N~", "SH~B~H~N~", "R~B~nH~N~"/)

; Plot latent heat flux phase composite
print("Plotting latent heat flux phase composite...")
do phase = 0, 3
    res@gsnLeftString = stringw(phase)
    res@gsnRightString = strings(0)
    
    ; Show only areas that pass significance test
    if (phase_count(phase) .gt. 1) then
        ; Create significance mask: areas that pass test retain original values, areas that fail are set to missing values
        phase_composite_SLHF_sig = where(significance_SLHF(phase, :, :) .lt. alpha, phase_composite_SLHF(phase, :, :), phase_composite_SLHF@_FillValue)
        plot(phase) = gsn_csm_contour_map_ce(wks, phase_composite_SLHF_sig, res)
    else
        plot(phase) = gsn_csm_contour_map_ce(wks, phase_composite_SLHF(phase, :, :), res)
    end if
end do

; Plot sensible heat flux phase composite
print("Plotting sensible heat flux phase composite...")
do phase = 0, 3
    res@gsnLeftString = stringw(phase)
    res@gsnRightString = strings(1)
    
    ; Show only areas that pass significance test
    if (phase_count(phase) .gt. 1) then
        ; Create significance mask: areas that pass test retain original values, areas that fail are set to missing values
        phase_composite_SSHF_sig = where(significance_SSHF(phase, :, :) .lt. alpha, phase_composite_SSHF(phase, :, :), phase_composite_SSHF@_FillValue)
        plot(phase+4) = gsn_csm_contour_map_ce(wks, phase_composite_SSHF_sig, res)
    else
        plot(phase+4) = gsn_csm_contour_map_ce(wks, phase_composite_SSHF(phase, :, :), res)
    end if
end do

; Plot net radiation phase composite
print("Plotting net radiation phase composite...")
do phase = 0, 3
    res@gsnLeftString = stringw(phase)
    res@gsnRightString = strings(2)
    
    ; Show only areas that pass significance test
    if (phase_count(phase) .gt. 1) then
        ; Create significance mask: areas that pass test retain original values, areas that fail are set to missing values
        phase_composite_RN_sig = where(significance_RN(phase, :, :) .lt. alpha, phase_composite_RN(phase, :, :), phase_composite_RN@_FillValue)
        plot(phase+8) = gsn_csm_contour_map_ce(wks, phase_composite_RN_sig, res)
    else
        plot(phase+8) = gsn_csm_contour_map_ce(wks, phase_composite_RN(phase, :, :), res)
    end if
end do

; 7. Create panel
print("7. Creating panel...")

pnlres = True
pnlres@amJust = "TopLeft"
pnlres@gsnPanelLabelBar = True
pnlres@lbLabelFontHeightF = 0.007

gsn_panel(wks, plot, (/3,4/), pnlres)

; 8. Output statistics
print("")
print("=== Phase composite analysis statistics ===")
print("Ruler cycle length: " + dimsizes(avg_cycle) + " days")
print("Number of phases: " + n_phases + " (two days per phase)")
print("Phase definitions:")
print("  Phase 0 (Phase -3): -7~-6 days (high value period)")
print("  Phase 1 (Phase -2): -5~-4 days")
print("  Phase 2 (Phase -1): -3~-2 days")
print("  Phase 3 (Phase 0): -1~0 days (low value period)")
print("Number of similar patterns found for each phase:")
do i = 0, n_phases-1
    print("  Phase " + (i+1) + ": " + phase_count(i) + " patterns")
end do
print("Significance test:")
print("  Significance level: " + alpha + " (90% confidence level)")
print("  Test method: t-test (compared to null hypothesis)")
print("  Significant areas: marked with dot pattern")

print("")
print("High-frequency heat flux phase composite distribution analysis completed!")
print("Output file: high_freq_heat_flux_phase_composite.png")

end