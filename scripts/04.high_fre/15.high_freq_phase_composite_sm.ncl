load "../../shapefiles/shapefile_utils.ncl"

begin

path = "../../data/climatology/filtered/"
print("=== High-frequency SM phase composite distribution analysis ===")

; 1. Read high-frequency SM average cycle as ruler
print("1. Reading high-frequency SM average cycle as ruler...")

f_avg_cycle = addfile(path+"high_freq_sm_avg_cycle_refined.nc", "r")
avg_cycle = f_avg_cycle->avg_cycle
cycle_time = f_avg_cycle->cycle_time
n_cycles = f_avg_cycle->n_cycles

print("Ruler cycle length: " + dimsizes(avg_cycle) + " days")
print("Number of typical cycles found: " + n_cycles)

; 2. Read high-frequency SM data
print("2. Reading high-frequency SM data...")

f_sml1_filtered = addfile(path+"mam_sml14icp_gms_filtered.nc", "r")
ts_high_SM = f_sml1_filtered->ts_high  ; (day, lat, lon)

; Read ERA5-land data for land-sea mask

pthl = "../../data/climatology/origin//"

f_era5 = addfile(pathl+"fmamj_sml14icp.nc", "r")
x = f_era5->sml1(28:119,:,:)
lsm = where(x.ne.x@_FillValue, 1, x@_FillValue)

; Get dimension information
lat = ts_high_SM&lat
lon = ts_high_SM&lon
nlat = dimsizes(lat)
nlon = dimsizes(lon)
ntime = dimsizes(ts_high_SM&day)

print("Data dimensions: " + ntime + " days, " + nlat + " latitude, " + nlon + " longitude")

; 3. Perform detrending
print("3. Performing detrending...")

; Detrend and standardize for each grid point
ts_high_SM_detrend = ts_high_SM

do i = 0, nlat-1
    do j = 0, nlon-1
        ; SM detrending
        if (.not. all(ismissing(ts_high_SM(:, i, j)))) then
            ts_high_SM_detrend(:, i, j) = dtrend(ts_high_SM(:, i, j), False)
            ts_high_SM_detrend(:, i, j) = dim_standardize_n_Wrap(ts_high_SM_detrend(:, i, j), 0, 0)
        end if
    end do
end do

; Apply land-sea mask
ts_high_SM_detrend = ts_high_SM_detrend * lsm

; 4. Apply shapefile mask
print("4. Applying shapefile mask...")

opt_mask = True
opt_mask@keep = True  ; Keep data within GMS region
opt_mask@return_mask = False

do d = 0, ntime-1
    ; Apply mask to each time slice
    sm_slice = ts_high_SM_detrend(d,:,:)
    sm_slice!0 = "lat"
    sm_slice!1 = "lon"
    
    ; Apply mask
    ts_high_SM_detrend(d,:,:) = shapefile_mask_data(sm_slice, "../../shapefiles/gms.shp", opt_mask)
end do

; 5. Phase composite based on ruler
print("5. Phase composite based on ruler...")

; Define number of phases (-7~0 days, two days per phase, 4 phases total)
n_phases = 4
phase_composite_SM = new((/n_phases, nlat, nlon/), float)
phase_composite_SM@_FillValue = 1e20

; Initialize counter
phase_count = new(n_phases, integer)
phase_count = 0

; Phase composite based on ruler's trough-peak-trough characteristics
; Phase definition: -7~-6 days as phase 0 (high value period), -5~-4 days as phase 1, -3~-2 days as phase 2, -1~0 days as phase 3 (low value period)
do phase = 0, n_phases-1
    print("  Processing phase " + (phase+1) + "/" + n_phases)
    
    ; Find patterns similar to ruler phase in time series
    do t = 1, ntime-15  ; Leave enough space
        ; Extract 15-day segment
        segment = ts_high_SM_detrend(t:t+14, :, :)
        
        ; Calculate similarity with ruler
        ; Calculate correlation coefficient (regional average)
        segment_avg = dim_avg_n_Wrap(segment(0:6, :, :), (/1,2/))
        corr = escorc(segment_avg, avg_cycle(0:6))
        
        ; If correlation coefficient is high enough, consider it a similar pattern
        if (corr .gt. 0.7) then
            ; Accumulate to phase composite
            if (phase_count(phase) .eq. 0) then
                ; Calculate corresponding day range based on phase (adjust order: high to low)
                if (phase .eq. 0) then
                    ; Phase 0: -7~-6 days, corresponding to ruler's day 7-8 (high value period)
                    phase_composite_SM(phase, :, :) = dim_avg_n_Wrap(segment(6:7, :, :), 0)
                else if (phase .eq. 1) then
                    ; Phase 1: -5~-4 days, corresponding to ruler's day 5-6
                    phase_composite_SM(phase, :, :) = dim_avg_n_Wrap(segment(4:5, :, :), 0)
                else if (phase .eq. 2) then
                    ; Phase 2: -3~-2 days, corresponding to ruler's day 3-4
                    phase_composite_SM(phase, :, :) = dim_avg_n_Wrap(segment(2:3, :, :), 0)
                else if (phase .eq. 3) then
                    ; Phase 3: -1~0 days, corresponding to ruler's day 1-2 (low value period)
                    phase_composite_SM(phase, :, :) = dim_avg_n_Wrap(segment(0:1, :, :), 0)
                end if
                end if
                end if
                end if
            else
                ; Accumulate pattern
                if (phase .eq. 0) then
                    phase_composite_SM(phase, :, :) = phase_composite_SM(phase, :, :) + dim_avg_n_Wrap(segment(6:7, :, :), 0)
                else if (phase .eq. 1) then
                    phase_composite_SM(phase, :, :) = phase_composite_SM(phase, :, :) + dim_avg_n_Wrap(segment(4:5, :, :), 0)
                else if (phase .eq. 2) then
                    phase_composite_SM(phase, :, :) = phase_composite_SM(phase, :, :) + dim_avg_n_Wrap(segment(2:3, :, :), 0)
                else if (phase .eq. 3) then
                    phase_composite_SM(phase, :, :) = phase_composite_SM(phase, :, :) + dim_avg_n_Wrap(segment(0:1, :, :), 0)
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
        phase_composite_SM(phase, :, :) = phase_composite_SM(phase, :, :) / phase_count(phase)
        print("    Phase " + (phase+1) + " found " + phase_count(phase) + " similar patterns")
    end if
end do

; 6. Perform significance test (t-test)
print("6. Performing significance test (t-test)...")

; Create significance test arrays
significance_SM = new((/n_phases, nlat, nlon/), float)
significance_SM@_FillValue = 1e20

; Set significance level (90% confidence level)
alpha = 0.10

; Perform t-test for each phase
do phase = 0, n_phases-1
    print("  Performing significance test for phase " + (phase+1) + "...")
    
    if (phase_count(phase) .gt. 1) then  ; At least 2 samples needed for t-test
        ; Re-collect all sample data for this phase
        phase_samples_SM = new((/phase_count(phase), nlat, nlon/), float)
        phase_samples_SM@_FillValue = 1e20
        
        sample_idx = 0
        
        ; Re-iterate through time series to collect samples
        do t = 1, ntime-15
            segment = ts_high_SM_detrend(t:t+14, :, :)
            segment_avg = dim_avg_n_Wrap(segment(0:6, :, :), (/1,2/))
            corr = escorc(segment_avg, avg_cycle(0:6))
            
            if (corr .gt. 0.7) then
                ; Extract corresponding data based on phase
                if (phase .eq. 0) then
                    phase_samples_SM(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_SM_detrend(t+6:t+7, :, :), 0)
                else if (phase .eq. 1) then
                    phase_samples_SM(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_SM_detrend(t+4:t+5, :, :), 0)
                else if (phase .eq. 2) then
                    phase_samples_SM(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_SM_detrend(t+2:t+3, :, :), 0)
                else if (phase .eq. 3) then
                    phase_samples_SM(sample_idx, :, :) = dim_avg_n_Wrap(ts_high_SM_detrend(t:t+1, :, :), 0)
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
                ; SM t-test
                if (.not. all(ismissing(phase_samples_SM(:, i, j)))) then
                    t_stat = ttest(phase_samples_SM(:, i, j), 0.0, phase_count(phase), 0.0, phase_count(phase), False, 0, 0)
                    significance_SM(phase, i, j) = t_stat(0)  ; p-value
                end if
            end do
        end do
        
        delete(phase_samples_SM)
    end if
end do

; 7. Plot phase composite distribution
print("7. Plotting phase composite distribution...")

wks = gsn_open_wks("png", "high_freq_sm_phase_composite")

; Create panel
plot = new(4, graphic)

; Basic plot resources (mimic script 10)
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

; Use color palette with warm color for negative anomalies and cool color for positive anomalies
FillColors = (/20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240/)
res@cnFillColors = FillColors(::-1)  ; Reverse color order

res@gsnLeftStringFontHeightF = 0.02

; Phase labels
strings = (/"(a)","(b)","(c)","(d)"/)
stringw = (/"Phase -3","Phase -2","Phase -1","Phase 0"/)

; Plot SM phase composite
print("Plotting SM phase composite...")
do phase = 0, 3
    res@gsnLeftString = strings(phase)+stringw(phase)
    res@gsnRightString = "SM~B~H~N~"
    
    ; Show only areas that pass significance test
    if (phase_count(phase) .gt. 1) then
        ; Create significance mask: areas that pass test retain original values, areas that fail are set to missing values
        phase_composite_SM_sig = where(significance_SM(phase, :, :) .lt. alpha, phase_composite_SM(phase, :, :), phase_composite_SM@_FillValue)
        plot(phase) = gsn_csm_contour_map_ce(wks, phase_composite_SM_sig, res)
    else
        plot(phase) = gsn_csm_contour_map_ce(wks, phase_composite_SM(phase, :, :), res)
    end if
end do

; 7. Create panel
print("7. Creating panel...")

pnlres = True
pnlres@amJust = "TopLeft"
pnlres@gsnPanelLabelBar = True
pnlres@lbLabelFontHeightF = 0.007

gsn_panel(wks, plot, (/1,4/), pnlres)

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
print("High-frequency SM phase composite distribution analysis completed!")
print("Output file: high_freq_sm_phase_composite.png")

end