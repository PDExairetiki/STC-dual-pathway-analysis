begin

print("=== High-frequency SM typical cycle analysis ===")

; 1. Read high-frequency SM data
print("1. Reading high-frequency SM data...")

path = "../../data/climatology/filtered/"

f_sml1_filtered = addfile(path+"mam_sml14icp_gms_filtered.nc", "r")
ts_high_SM = f_sml1_filtered->ts_high  ; (day, lat, lon)

; Get dimension information
dims = dimsizes(ts_high_SM)
ntime = dims(0)
nlat = dims(1)
nlon = dims(2)

print("Data dimensions: " + ntime + " days, " + nlat + " latitude, " + nlon + " longitude")

; 2. Calculate regional average high-frequency SM time series and perform detrending
print("2. Calculating regional average high-frequency SM time series and performing detrending...")

; Calculate regional average (removing NaN values)
ts_high_SM_avg = dim_avg_n_Wrap(ts_high_SM, (/1,2/))  ; (day)

; Seasonal detrending (92 days)
print("  Performing seasonal detrending...")
ts_high_SM_avg = dtrend(ts_high_SM_avg, False)

; Standardization processing
print("  Performing standardization...")
ts_high_SM_avg = dim_standardize_n_Wrap(ts_high_SM_avg, 0, 0)

; 3. Define typical cycle conditions
print("3. Defining typical cycle conditions...")

; Peak greater than 1 standard deviation, trough less than -1 standard deviation
peak_threshold = 1.0
trough_threshold = -1.0

; 4. Find typical cycles that meet conditions
print("4. Finding typical cycles that meet conditions...")

; Initialize array to store found cycles
max_cycles = 6  ; Store up to 6 cycles
found_cycles = 0
cycle_data = new((/max_cycles, 15/), float)  ; Fixed 15-day length
cycle_data@_FillValue = 1e20

; Find peaks and troughs
peaks = new(ntime, integer)
troughs = new(ntime, integer)
n_peaks = 0
n_troughs = 0

; Find peaks
do i = 1, ntime-2
    if (ts_high_SM_avg(i) .gt. peak_threshold .and. \
        ts_high_SM_avg(i) .gt. ts_high_SM_avg(i-1) .and. \
        ts_high_SM_avg(i) .gt. ts_high_SM_avg(i+1)) then
        peaks(n_peaks) = i
        n_peaks = n_peaks + 1
    end if
end do

; Find troughs
do i = 1, ntime-2
    if (ts_high_SM_avg(i) .lt. trough_threshold .and. \
        ts_high_SM_avg(i) .lt. ts_high_SM_avg(i-1) .and. \
        ts_high_SM_avg(i) .lt. ts_high_SM_avg(i+1)) then
        troughs(n_troughs) = i
        n_troughs = n_troughs + 1
    end if
end do

print("Found " + n_peaks + " peaks, " + n_troughs + " troughs")

; Find complete "trough-peak-trough" cycles
do i = 0, n_troughs-2  ; Iterate through troughs
    if (found_cycles .ge. max_cycles) then
        break
    end if
    
    trough1 = troughs(i)
    trough2 = troughs(i+1)
    
    ; Find peak between two troughs
    peak_found = False
    peak_time = -1
    
    do j = 0, n_peaks-1
        if (peaks(j) .gt. trough1 .and. peaks(j) .lt. trough2) then
            peak_found = True
            peak_time = peaks(j)
            break
        end if
    end do
    
    ; If complete cycle found
    if (peak_found) then
        cycle_len = trough2 - trough1 + 1
        
        ; Check if cycle length is reasonable (10-20 days)
        if (cycle_len .ge. 10 .and. cycle_len .le. 20) then
            ; Extract cycle data
            if (trough1 + 14 .le. ntime-1) then
                cycle_segment = ts_high_SM_avg(trough1:trough1+14)
                
                ; Store cycle
                cycle_data(found_cycles, :) = cycle_segment
                found_cycles = found_cycles + 1
                
                print("Found cycle " + found_cycles + ", time range: " + trough1 + "-" + (trough1+14))
                print("  Starting trough: " + sprintf("%6.3f", ts_high_SM_avg(trough1)))
                print("  Peak: " + sprintf("%6.3f", ts_high_SM_avg(peak_time)))
                print("  Ending trough: " + sprintf("%6.3f", ts_high_SM_avg(trough2)))
                print("  Cycle length: " + cycle_len + " days")
            end if
        end if
    end if
end do

print("Total found " + found_cycles + " typical cycles")

; 5. Calculate average cycle
print("5. Calculating average cycle...")

if (found_cycles .gt. 0) then
    ; Calculate average cycle
    avg_cycle = dim_avg_n_Wrap(cycle_data(0:found_cycles-1, :), 0)
    
    ; Calculate standard deviation
    std_cycle = dim_stddev_n_Wrap(cycle_data(0:found_cycles-1, :), 0)
    
    print("Average cycle calculation completed")
    print("Average cycle peak: " + sprintf("%6.3f", max(avg_cycle)))
    print("Average cycle trough: " + sprintf("%6.3f", min(avg_cycle)))
else
    print("No typical cycles meeting conditions found")
    exit
end if

; 6. Create time coordinates
print("6. Creating time coordinates...")

cycle_time = ispan(0, 14, 1)
cycle_time@long_name = "Day within cycle"
cycle_time@units = "days"

; 7. Plot results
print("7. Plotting results...")

wks = gsn_open_wks("png", "high_freq_sm_cycle_refined")

; Create panel
plot = new(2, graphic)

; Basic plot resources
res = True
res@gsnDraw = False
res@gsnFrame = False
res@gsnMaximize = True

; Set figure size
res@vpHeightF = 0.3
res@vpWidthF = 0.6

res@tmXTOn = False
res@tmYROn = False

; Axis settings
res@tmXMajorGrid = True
res@tmYMajorGrid = True
res@tmXMajorGridLineColor = "gray80"
res@tmYMajorGridLineColor = "gray80"
res@tmXMajorGridThicknessF = 0.5
res@tmYMajorGridThicknessF = 0.5

res@tmXMinorGrid = True
res@tmYMinorGrid = True
res@tmXMinorGridLineColor = "gray90"
res@tmYMinorGridLineColor = "gray90"
res@tmXMinorGridThicknessF = 0.25
res@tmYMinorGridThicknessF = 0.25

; Zero line settings
res@gsnYRefLine = 0.0
res@gsnYRefLineColor = "gray50"
res@gsnYRefLineThicknessF = 1.0
res@gsnYRefLineDashPattern = 2

; Set Y-axis range
res@trYMinF = -3
res@trYMaxF = 3

; Set X-axis labels
res@tmXBMode = "Explicit"
res@tmXBValues = ispan(0, 14, 2)
res@tmXBLabels = ispan(0, 14, 2)

; Plot all found cycles
print("Plotting all found cycles...")

res@xyLineColors = "gray"
res@xyLineThicknesses = 0.5
res@tiYAxisString = "Amplitude"
res@xyDashPatterns = 0

; Plot each cycle
res@gsnLeftString = "(a) All Found Cycles"
res@gsnRightString = "High-frequency SM Cycles"

; First plot the first cycle
plot(0) = gsn_csm_xy(wks, cycle_time, cycle_data(0, :), res)

; Then overlay other cycles
do i = 1, found_cycles-1
    overlay(plot(0), gsn_csm_xy(wks, cycle_time, cycle_data(i, :), res))
end do

; Plot average cycle
print("Plotting average cycle...")

res@xyLineColors = "red"
res@xyLineThicknesses = 3.0
res@xyDashPatterns = 0

res@gsnLeftString = "(b) Average Cycle"
res@gsnRightString = "Mean High-frequency SM Cycle"
plot(1) = gsn_csm_xy(wks, cycle_time, avg_cycle, res)

; Add error range (standard deviation)
if (found_cycles .gt. 1) then
    ; Calculate upper and lower error bounds
    upper_bound = avg_cycle + std_cycle
    lower_bound = avg_cycle - std_cycle
    
    ; Create error range array
    error_range = new((/2, 15/), float)
    error_range(0, :) = upper_bound
    error_range(1, :) = lower_bound
    
    ; Plot error range
    res_error = True
    res_error@gsnDraw = False
    res_error@gsnFrame = False
    res_error@xyLineColors = "red"
    res_error@xyLineThicknesses = 1.0
    res_error@xyDashPatterns = 2
    
    overlay(plot(1), gsn_csm_xy(wks, cycle_time, error_range, res_error))
end if

; 8. Create panel
print("8. Creating panel...")

pres = True
pres@gsnPanelMainFontHeightF = 0.025
pres@gsnPanelMainFont = "helvetica-bold"
pres@gsnMaximize = True
pres@gsnPaperOrientation = "portrait"
pres@gsnPanelRowSpec = True
pres@gsnPanelLabelBar = False

gsn_panel(wks, plot, (/2,1/), pres)

; 9. Save average cycle data
print("9. Saving average cycle data...")

; Create output file
system("rm -f high_freq_sm_avg_cycle_refined.nc")
fout = addfile("high_freq_sm_avg_cycle_refined.nc", "c")

; Set file attributes
fout@title = "High-frequency SM Average Cycle"
fout@source = "Extracted from mam_sml14icp_gms_filtered.nc"
fout@creation_date = systemfunc("date")

; Write data
fout->avg_cycle = avg_cycle
fout->std_cycle = std_cycle
fout->cycle_time = cycle_time
fout->n_cycles = found_cycles

; 10. Output statistics
print("")
print("=== High-frequency SM cycle statistics ===")
print("Number of typical cycles found: " + found_cycles)
print("Peak threshold: > " + peak_threshold + " standard deviations")
print("Trough threshold: < " + trough_threshold + " standard deviations")

print("")
print("Average cycle characteristics:")
print("  Average peak: " + sprintf("%6.3f", max(avg_cycle)))
print("  Average trough: " + sprintf("%6.3f", min(avg_cycle)))
print("  Average amplitude: " + sprintf("%6.3f", max(avg_cycle) - min(avg_cycle)))

if (found_cycles .gt. 1) then
    print("  Standard deviation within cycles: " + sprintf("%6.3f", avg(std_cycle)))
end if

print("")
print("High-frequency SM typical cycle analysis completed!")
print("Output file: high_freq_sm_cycle_refined.png")
print("Data file: high_freq_sm_avg_cycle_refined.nc")

end