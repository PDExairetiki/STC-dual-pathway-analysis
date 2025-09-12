load "../../shapefiles/shapefile_utils.ncl"

begin

print("=== High-frequency T and STC phase composite distribution analysis ===")

; 1. Read high-frequency SM average cycle as ruler
print("1. Reading high-frequency SM average cycle as ruler...")

path = "../../data/climatology/filtered/"

f_avg_cycle = addfile(path+"high_freq_sm_avg_cycle_refined.nc", "r")
avg_cycle = f_avg_cycle->avg_cycle
cycle_time = f_avg_cycle->cycle_time
n_cycles = f_avg_cycle->n_cycles

print("Ruler cycle length: " + dimsizes(avg_cycle) + " days")
print("Number of typical cycles found: " + n_cycles)

; 2. Read high-frequency T and STC data
print("2. Reading high-frequency T and STC data...")

; Read high-frequency T2M data
f_t2m_filtered = addfile(path+"mam_t2m4icp_gms_filtered.nc", "r")
ts_high_T = f_t2m_filtered->ts_high  ; (day, lat, lon)

; Read high-frequency STC data
f_paip_filtered = addfile(path+"mam_paip4icp_gms_filtered.nc", "r")
ts_high_STC = f_paip_filtered->ts_high  ; (day, lat, lon)

; Read ERA5-land data for land-sea mask

pathl = "../../data/climatology/origin/"

f_era5_t = addfile("fmamj_t2m4icp.nc", "r")
x_t = f_era5_t->t2m(28:119,:,:)
lsm_t = where(x_t.ne.x_t@_FillValue, 1, x_t@_FillValue)

f_era5_stc = addfile("fmamj_paip4icp.nc", "r")
x_stc = f_era5_stc->paip(28:119,:,:)
lsm_stc = where(x_stc.ne.x_stc@_FillValue, 1, x_stc@_FillValue)

; Get dimension information
lat = ts_high_T&lat
lon = ts_high_T&lon
nlat = dimsizes(lat)
nlon = dimsizes(lon)
ntime = dimsizes(ts_high_T&day)

print("Data dimensions: " + ntime + " days, " + nlat + " latitude, " + nlon + " longitude")

; 3. Perform detrending
print("3. Performing detrending...")

; Detrend and standardize for each grid point
ts_high_T_detrend = ts_high_T
ts_high_STC_detrend = ts_high_STC

do i = 0, nlat-1
    do j = 0, nlon-1
        ; T detrending
        if (.not. all(ismissing(ts_high_T(:, i, j)))) then
            ts_high_T_detrend(:, i, j) = dtrend(ts_high_T(:, i, j), False)
            ts_high_T_detrend(:, i, j) = dim_standardize_n_Wrap(ts_high_T_detrend(:, i, j), 0, 0)
        end if
        
        ; STC detrending
        if (.not. all(ismissing(ts_high_STC(:, i, j)))) then
            ts_high_STC_detrend(:, i, j) = dtrend(ts_high_STC(:, i, j), False)
            ts_high_STC_detrend(:, i, j) = dim_standardize_n_Wrap(ts_high_STC_detrend(:, i, j), 0, 0)
        end if
    end do
end do

; Apply land-sea mask
ts_high_T_detrend = ts_high_T_detrend * lsm_t
ts_high_STC_detrend = ts_high_STC_detrend * lsm_stc

; 4. Apply shapefile mask
print("4. Applying shapefile mask...")

opt_mask = True
opt_mask@keep = True  ; Keep data within GMS region
opt_mask@return_mask = False

do d = 0, ntime-1
    ; Apply mask to each time slice
    t_slice = ts_high_T_detrend(d,:,:)
    t_slice!0 = "lat"
    t_slice!1 = "lon"
    
    stc_slice = ts_high_STC_detrend(d,:,:)
    stc_slice!0 = "lat"
    stc_slice!1 = "lon"
    
    ; Apply mask
    ts_high_T_detrend(d,:,:) = shapefile_mask_data(t_slice, "../../shapefiles/gms.shp", opt_mask)
    ts_high_STC_detrend(d,:,:) = shapefile_mask_data(stc_slice, "../../shapefiles/gms.shp", opt_mask)
end do

; 5. Phase composite based on ruler
print("5. Phase composite based on ruler...")

; Define number of phases (-7~0 days, two days per phase, 4 phases total)
n_phases = 4
phase_composite_T = new((/n_phases, nlat, nlon/), float)
phase_composite_STC = new((/n_phases, nlat, nlon/), float)
phase_composite_T@_FillValue = 1e20
phase_composite_STC@_FillValue = 1e20

; Initialize counter
phase_count = new(n_phases, integer)
phase_count = 0

; Phase composite based on ruler's trough-peak-trough characteristics
; Phase definition: -7~-6 days as phase 0 (high value period), -5~-4 days as phase 1, -3~-2 days as phase 2, -1~0 days as phase 3 (low value period)
do phase = 0, n_phases-1
    print("  Processing phase " + (phase+1) + "/" + n_phases)
    
    ; Find patterns similar to ruler phase in time series
    do t = 1, ntime-15  ; Leave enough space
        ; Extract 15-day segment (use SM data to calculate similarity)
        segment = ts_high_T_detrend(t:t+14, :, :)
        
        ; Calculate similarity with ruler
        ; Calculate correlation coefficient (regional average)
        segment_avg = dim_avg_n_Wrap(segment(0:6, :, :), (/1,2/))
        corr = escorc(segment_avg, avg_cycle(0:6))
        
        ; If correlation coefficient is high enough, consider it a similar pattern
        if (corr .gt. 0.7) then
            ; Accumulate to phase composite
            if (phase_count(phase) .eq. 0) then
                ; Calculate corresponding day range based on phase
                if (phase .eq. 0) then
                    ; Phase 0: -7~-6 days, corresponding to ruler's day 1-2
                    phase_composite_T(phase, :, :) = dim_avg_n_Wrap(ts_high_T_detrend(t:t+1, :, :), 0)
                    phase_composite_STC(phase, :, :) = dim_avg_n_Wrap(ts_high_STC_detrend(t:t+1, :, :), 0)
                else if (phase .eq. 1) then
                    ; Phase 1: -5~-4 days, corresponding to ruler's day 3-4
                    phase_composite_T(phase, :, :) = dim_avg_n_Wrap(ts_high_T_detrend(t+2:t+3, :, :), 0)
                    phase_composite_STC(phase, :, :) = dim_avg_n_Wrap(ts_high_STC_detrend(t+2:t+3, :, :), 0)
                else if (phase .eq. 2) then
                    ; Phase 2: -3~-2 days, corresponding to ruler's day 5-6
                    phase_composite_T(phase, :, :) = dim_avg_n_Wrap(ts_high_T_detrend(t+4:t+5, :, :), 0)
                    phase_composite_STC(phase, :, :) = dim_avg_n_Wrap(ts_high_STC_detrend(t+4:t+5, :, :), 0)
                else if (phase .eq. 3) then
                    ; Phase 3: -1~0 days, corresponding to ruler's day 7-8
                    phase_composite_T(phase, :, :) = dim_avg_n_Wrap(ts_high_T_detrend(t+6:t+7, :, :), 0)
                    phase_composite_STC(phase, :, :) = dim_avg_n_Wrap(ts_high_STC_detrend(t+6:t+7, :, :), 0)
                end if
                end if
                end if
                end if
            else
                ; Accumulate pattern
                if (phase .eq. 0) then
                    phase_composite_T(phase, :, :) = phase_composite_T(phase, :, :) + dim_avg_n_Wrap(ts_high_T_detrend(t:t+1, :, :), 0)
                    phase_composite_STC(phase, :, :) = phase_composite_STC(phase, :, :) + dim_avg_n_Wrap(ts_high_STC_detrend(t:t+1, :, :), 0)
                else if (phase .eq. 1) then
                    phase_composite_T(phase, :, :) = phase_composite_T(phase, :, :) + dim_avg_n_Wrap(ts_high_T_detrend(t+2:t+3, :, :), 0)
                    phase_composite_STC(phase, :, :) = phase_composite_STC(phase, :, :) + dim_avg_n_Wrap(ts_high_STC_detrend(t+2:t+3, :, :), 0)
                else if (phase .eq. 2) then
                    phase_composite_T(phase, :, :) = phase_composite_T(phase, :, :) + dim_avg_n_Wrap(ts_high_T_detrend(t+4:t+5, :, :), 0)
                    phase_composite_STC(phase, :, :) = phase_composite_STC(phase, :, :) + dim_avg_n_Wrap(ts_high_STC_detrend(t+4:t+5, :, :), 0)
                else if (phase .eq. 3) then
                    phase_composite_T(phase, :, :) = phase_composite_T(phase, :, :) + dim_avg_n_Wrap(ts_high_T_detrend(t+6:t+7, :, :), 0)
                    phase_composite_STC(phase, :, :) = phase_composite_STC(phase, :, :) + dim_avg_n_Wrap(ts_high_STC_detrend(t+6:t+7, :, :), 0)
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
        phase_composite_T(phase, :, :) = phase_composite_T(phase, :, :) / phase_count(phase)
        phase_composite_STC(phase, :, :) = phase_composite_STC(phase, :, :) / phase_count(phase)
        print("    Phase " + (phase+1) + " found " + phase_count(phase) + " similar patterns")
    end if
end do

; 6. Perform significance test (t-test)
print("6. Performing significance test (t-test)...")

; Create significance test arrays
significance_T = new((/n_phases, nlat, nlon/), float)
significance_STC = new((/n_phases, nlat, nlon/), float)
significance_T@_FillValue = 1e20
significance_STC@_FillValue = 1e20

; Set significance level (90% confidence level)
alpha = 0.10

; Perform t-test for each phase
do phase = 0, n_phases-1
    print("  Performing significance test for phase " + (phase+1) + "...")
    
    if (phase_count(phase) .gt. 1) then  ; At least 2 samples needed for t-test
        ; Re-collect all sample data for this phase
        phase_samples_T = new((/phase_count(phase), nlat, nlon/), float)
        phase_samples_STC = new((/phase_count(phase), nlat, nlon/), float)
        phase_samples_T@_FillValue = 1e20
        phase_samples_STC@_FillValue = 1e20
        
        sample_idx = 0
        
        ; Re-iterate through time series to collect samples
        do t = 1, ntime-15
            segment = ts_high_T_detrend(t:t+14, :, :)
            segment_avg = dim_avg_n_Wrap(segment(0:6, :, :), (/1,2/))
            corr = escorc(segment_avg, avg_cycle(0:6))
            
            if (corr .gt. 0.7) then
                ; Extract corresponding data based on phase
                if (phase .eq. 0) then
                    phase_samples_T(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_T_detrend(t+6:t+7, :, :), 0)
                    phase_samples_STC(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_STC_detrend(t+6:t+7, :, :), 0)
                else if (phase .eq. 1) then
                    phase_samples_T(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_T_detrend(t+4:t+5, :, :), 0)
                    phase_samples_STC(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_STC_detrend(t+4:t+5, :, :), 0)
                else if (phase .eq. 2) then
                    phase_samples_T(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_T_detrend(t+2:t+3, :, :), 0)
                    phase_samples_STC(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_STC_detrend(t+2:t+3, :, :), 0)
                else if (phase .eq. 3) then
                    phase_samples_T(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_T_detrend(t:t+1, :, :), 0)
                    phase_samples_STC(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_STC_detrend(t:t+1, :, :), 0)
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
        
        ; Perform t-test for each grid point
        do i = 0, nlat-1
            do j = 0, nlon-1
                ; T t-test
                if (.not. all(ismissing(phase_samples_T(:, i, j)))) then
                    t_stat = ttest(phase_samples_T(:, i, j), 0.0, phase_count(phase), 0.0, phase_count(phase), False, 0, 0)
                    significance_T(phase, i, j) = t_stat(0)  ; p-value
                end if
                
                ; STC t-test
                if (.not. all(ismissing(phase_samples_STC(:, i, j)))) then
                    t_stat = ttest(phase_samples_STC(:, i, j), 0.0, phase_count(phase), 0.0, phase_count(phase), False, 0, 0)
                    significance_STC(phase, i, j) = t_stat(0)  ; p-value
                end if
            end do
        end do
        
        delete(phase_samples_T)
        delete(phase_samples_STC)
    end if
end do

; 7. Plot phase composite distribution
print("7. Plotting phase composite distribution...")

wks = gsn_open_wks("png", "high_freq_t_stc_phase_composite")

; Create panel

plot = new(8, graphic)

; Basic plot resources (mimic script 9)
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

; Color settings: warm color for negative anomalies, cool color for positive anomalies
res@cnLevelSelectionMode = "ManualLevels"
res@cnMinLevelValF = -1.0
res@cnMaxLevelValF = 1.0
res@cnLevelSpacingF = 0.1

; Use cool color for negative anomalies, warm color for positive anomalies
FillColors = (/20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240/)
res@cnFillColors = FillColors  ; Do not reverse color order

res@gsnLeftStringFontHeightF = 0.02

; Phase labels
stringl = (/"(e)","(f)","(g)","(h)","(i)","(j)","(k)","(l)","(m)","(n)"/)
stringw = (/"Phase -3", "Phase -2", "Phase -1", "Phase 0"/)
strings = (/"T~B~2mH~N~", "STC - ~F33~p~FB~H~N~"/)

; Plot T phase composite
print("Plotting T phase composite...")
do phase = 0, 3
    res@gsnLeftString = stringl(phase)+stringw(phase)
    res@gsnRightString = strings(0)
    
    ; Show only areas that pass significance test
    if (phase_count(phase) .gt. 1) then
        ; Create significance mask: areas that pass test retain original values, areas that fail are set to missing values
        phase_composite_T_sig = where(significance_T(phase, :, :) .lt. alpha, phase_composite_T(phase, :, :), phase_composite_T@_FillValue)
        plot(phase) = gsn_csm_contour_map_ce(wks, phase_composite_T_sig, res)
    else
        plot(phase) = gsn_csm_contour_map_ce(wks, phase_composite_T(phase, :, :), res)
    end if
end do

; Plot STC phase composite
print("Plotting STC phase composite...")
do phase = 0, 3
    res@gsnLeftString = stringl(phase+4)+stringw(phase)
    res@gsnRightString = strings(1)
    
    ; Show only areas that pass significance test
    if (phase_count(phase) .gt. 1) then
        ; Create significance mask: areas that pass test retain original values, areas that fail are set to missing values
        phase_composite_STC_sig = where(significance_STC(phase, :, :) .lt. alpha, phase_composite_STC(phase, :, :), phase_composite_STC@_FillValue)
        plot(phase+4) = gsn_csm_contour_map_ce(wks, phase_composite_STC_sig, res)
    else
        plot(phase+4) = gsn_csm_contour_map_ce(wks, phase_composite_STC(phase, :, :), res)
    end if
end do

; 7. Create panel
print("7. Creating panel...")

pnlres = True
pnlres@amJust = "TopLeft"
pnlres@gsnPanelLabelBar = True
pnlres@lbLabelFontHeightF = 0.007

gsn_panel(wks, plot, (/2,4/), pnlres)

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
print("High-frequency T and STC phase composite distribution analysis completed!")
print("Output file: high_freq_t_stc_phase_composite.png")

end