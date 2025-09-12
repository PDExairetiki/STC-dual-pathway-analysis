begin

path0="../../data/climatology/eof/"
path1="../../data/climatology/filtered/"

print("=== Plot PC1 Time Series and Filtered Features ===")

; 1. Read PC1 time series data
print("1. Reading PC1 time series data...")

; Read SML1 PC1
f_sml1_pc1 = addfile(path0+"mam_sml12ts4icp_gms.nc", "r")
ts_sm0 = f_sml1_pc1->ts(:,0,:)  ; Take first mode for all years
pc1_sm = dim_avg_n_Wrap(ts_sm0, 0)  ; Average across years
pc1_sm = (-1)*dim_standardize_n_Wrap(pc1_sm,0,0)

; Read T2M PC1  
f_t2m_pc1 = addfile(path1+"mam_t2m2ts4icp_gms.nc", "r")
ts_t2m0 = f_t2m_pc1->ts(:,0,:)  ; Take first mode for all years
pc1_t2m = dim_avg_n_Wrap(ts_t2m0, 0)  ; Average across years
pc1_t2m = dim_standardize_n_Wrap(pc1_t2m,0,0)

; 2. Read filtered high and low frequency data
print("2. Reading filtered high and low frequency data...")

; Read SML1 filtering results
f_sml1_filtered = addfile(path1+"mam_sml12ts4icp_gms_filtered.nc", "r")
ts_low_SM = f_sml1_filtered->ts_low             ; Low frequency SM (phase flipped in step 06)
ts_high_SM = f_sml1_filtered->ts_high           ; High frequency SM (phase flipped in step 06)
ts_low_SM = dtrend(ts_low_SM,False)
ts_low_SM = dim_standardize_n_Wrap(ts_low_SM,0,0)
ts_high_SM=dtrend(ts_high_SM,False)
ts_high_SM = dim_standardize_n_Wrap(ts_high_SM,0,0)


; Read T2M filtering results
f_t2m_filtered = addfile(path1+"mam_t2m2ts4icp_gms_filtered.nc", "r")
ts_low_T = f_t2m_filtered->ts_low      ; Low frequency T
ts_high_T = f_t2m_filtered->ts_high    ; High frequency T
ts_low_T=dtrend(ts_low_T,False)
ts_low_T = dim_standardize_n_Wrap(ts_low_T,0,0)
ts_high_T=dtrend(ts_high_T,False)
ts_high_T = dim_standardize_n_Wrap(ts_high_T,0,0)

; 3. Create time coordinates
print("3. Creating time coordinates...")

ntime = dimsizes(pc1_sm)
time = ispan(1, ntime, 1)
time@long_name = "Day"
time@units = "days since March 1"

; 4. Set plot resources
print("4. Setting plot resources...")

wks = gsn_open_wks("png", "pc1_time_series_analysis")

; Set panel
plot = new(2, graphic)

; Basic plot resources
res = True
res@gsnDraw = False
res@gsnFrame = False
res@gsnMaximize = True

; Set figure size to half width
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

; Set Y-axis range to ensure high frequency components don't exceed
res@trYMinF = -3
res@trYMaxF = 3

; Set X-axis labels
res@tmXBMode = "Explicit"
res@tmXBValues = (/0, 14, 31, 45, 60, 74/)
res@tmXBLabels = (/"Mar", "15th", "Apr", "15th", "May", "15th"/)

res@xyLineColors = (/"red","blue","green"/)
res@xyLineThicknesses = (/3.0,3.0,3.0/)
res@tiYAxisString = "Amplitude"
res@xyDashPatterns = (/0,0,0/)

; 5. Plot SM PC1 and high/low frequency components
print("5. Plotting SM PC1 and high/low frequency components...")

; Plot original PC1 (red solid line)
tss=(/pc1_sm,ts_low_SM,ts_high_SM/)
res@gsnLeftString="(c) MAM"
res@gsnRightString = "SM-PC1 & Frequency Components"
plot(0) = gsn_csm_xy(wks, time, tss, res)

; 6. Plot T2M PC1 and high/low frequency components
print("6. Plotting T2M PC1 and high/low frequency components...")

; Plot original PC1 (red solid line)
tst=(/pc1_t2m,ts_low_T,ts_high_T/)
res@gsnLeftString="(d) MAM"
res@gsnRightString = "T-PC1 & Frequency Components"
plot(1) = gsn_csm_xy(wks, time, tst, res)

; 7. Create legend
print("7. Creating legend...")

res_text=True
res_text@txFontHeightF=0.015
res_text@txJust="CenterLeft"

res_lines=True 
res_lines@gsLineDashPattern=0
res_lines@gsLineThicknessF=3.

res_li0=res_lines
res_li1=res_lines
res_li2=res_lines
res_li3=res_lines

res_li0@gsLineColor="red"
res_li1@gsLineColor="blue"
res_li2@gsLineColor="green"

xx0=(/40,45/)
xx1=(/60,65/)
xx2=(/75,80/);(/10,15/);
yy=(/-2.5,-2.5/);

dums001=gsn_add_polyline(wks,plot(0),xx0,yy,res_li0)
dums002=gsn_add_text(wks,plot(0),"SM-PC1",47,yy,res_text)
dums011=gsn_add_polyline(wks,plot(0),xx1,yy,res_li1)
dums012=gsn_add_text(wks,plot(0),"SM~B~L~N~",67,yy,res_text)
dums021=gsn_add_polyline(wks,plot(0),xx2,yy,res_li2)
dums022=gsn_add_text(wks,plot(0),"SM~B~H~N~",82,yy,res_text)

dums101=gsn_add_polyline(wks,plot(1),xx0,yy,res_li0)
dums102=gsn_add_text(wks,plot(1),"T-PC1",47,yy,res_text)
dums111=gsn_add_polyline(wks,plot(1),xx1,yy,res_li1)
dums112=gsn_add_text(wks,plot(1),"T~B~L~N~",67,yy,res_text)
dums121=gsn_add_polyline(wks,plot(1),xx2,yy,res_li2)
dums122=gsn_add_text(wks,plot(1),"T~B~H~N~",82,yy,res_text)

; 8. Create panel
print("8. Creating panel...")

pres = True
; pres@gsnPanelMainString = "PC1 Time Series Analysis: SM and T2M"
pres@gsnPanelMainFontHeightF = 0.025
pres@gsnPanelMainFont = "helvetica-bold"
pres@gsnMaximize = True
pres@gsnPaperOrientation = "portrait"
pres@gsnPanelRowSpec = True
pres@gsnPanelLabelBar = False

; Set panel size, ensure height is half of width
; pres@gsnPanelWidthF = 0.8
; pres@gsnPanelHeightF = 0.4

gsn_panel(wks, plot, (/2,1/), pres)

; 9. Calculate statistics
print("9. Calculating statistics...")

print("")
print("=== PC1 Time Series Statistics ===")
print("SM PC1:")
print("  Mean: " + sprintf("%6.3f", avg(pc1_sm)))
print("  Standard deviation: " + sprintf("%6.3f", stddev(pc1_sm)))
print("  Maximum: " + sprintf("%6.3f", max(pc1_sm)))
print("  Minimum: " + sprintf("%6.3f", min(pc1_sm)))

print("T2M PC1:")
print("  Mean: " + sprintf("%6.3f", avg(pc1_t2m)))
print("  Standard deviation: " + sprintf("%6.3f", stddev(pc1_t2m)))
print("  Maximum: " + sprintf("%6.3f", max(pc1_t2m)))
print("  Minimum: " + sprintf("%6.3f", min(pc1_t2m)))

print("SM low frequency component:")
print("  Mean: " + sprintf("%6.3f", avg(ts_low_SM)))
print("  Standard deviation: " + sprintf("%6.3f", stddev(ts_low_SM)))

print("SM high frequency component:")
print("  Mean: " + sprintf("%6.3f", avg(ts_high_SM)))
print("  Standard deviation: " + sprintf("%6.3f", stddev(ts_high_SM)))

print("T2M low frequency component:")
print("  Mean: " + sprintf("%6.3f", avg(ts_low_T)))
print("  Standard deviation: " + sprintf("%6.3f", stddev(ts_low_T)))

print("T2M high frequency component:")
print("  Mean: " + sprintf("%6.3f", avg(ts_high_T)))
print("  Standard deviation: " + sprintf("%6.3f", stddev(ts_high_T)))

print("")
print("PC1 time series analysis completed!")
print("Output file: pc1_time_series_analysis.png")

end