begin

path = "../../data/2019/eof/"

; Read SM and T PC1 time series
sm_file = addfile(path+"2019_sm_ts.nc","r")
t_file = addfile(path+"2019_t_ts.nc","r")

ts_sm0 = sm_file->ts  ; SM PC1 time series
ts_t0 = t_file->ts    ; T PC1 time series

; Detrend
ts_sm0 = dtrend_n(ts_sm0, False, 0)
ts_t0 = dtrend_n(ts_t0 ,False, 0)

ts_sm = dim_avg_n_Wrap(ts_sm0, 0)
ts_t = dim_avg_n_Wrap(ts_t0, 0)

ts_sm = dtrend_n(ts_sm, False, 0)
ts_t = dtrend_n(ts_t, False, 0)

print("SM PC1 time series length: " + dimsizes(ts_sm))
print("T PC1 time series length: " + dimsizes(ts_t))

; Cross-spectrum analysis parameter settings
iopt = 1      ; Use Hanning window
jave = 3      ; Average window size
pct = 0.10    ; Confidence level

; Calculate cross-spectrum
csdof = specxy_anal(ts_sm, ts_t, iopt, jave, pct)

print("Frequency range: " + min(csdof@frq) + " to " + max(csdof@frq))
print("Coherence range: " + min(csdof@coher) + " to " + max(csdof@coher))
print("Phase range: " + min(csdof@phase) + " to " + max(csdof@phase))

; Plot settings
wks = gsn_open_wks("png","2019_cpsd_sm_t")

stringw = (/"(a)","(b)"/)
strings = (/"Heatwave"/)
plots = new(2,graphic)

; Basic plot resources
rts = True
rts@gsnDraw = False
rts@gsnFrame = False
rts@gsnScale = True

rts@xyLineThicknessF = (/3,3/)
rts@xyLineColors = (/"black","red"/)
rts@xyDashPatterns = (/0,2/)

rts@tiYAxisString = ""

rts@tmXTOn = False
rts@tmYROn = False
rts@tmXBLabelFontHeightF = 0.02
rts@tmYLLabelFontHeightF = 0.02

rts@gsnLeftStringFontHeightF = 0.02

rts@vpHeightF = 0.3
rts@vpWidthF = 0.6

; Coherence plot
rts0 = rts
rts0@gsnYRefLine = 0.32
rts0@gsnYRefLineColor = "black"
rts0@gsnYRefLineDashPattern = 2
rts0@gsnAboveYRefLineColor = "red"
rts0@trYMinF = 0.
rts0@trYMaxF = 1.
rts0@gsnLeftString = stringw(0) + " " + strings(0)
rts0@gsnRightString = "Coherence(SM, T)"
rts0@tiXAxisString = "Period (days)"
; rts0@tiYAxisString = "Coherence"
plots(0) = gsn_csm_xy(wks, 1./csdof@frq, csdof@coher, rts0)

; Phase plot
rts1 = rts
rts1@gsnLeftString = stringw(1) + " " + strings(0)
rts1@gsnRightString = "Phase(SM, T)"
rts1@gsnYRefLine = 0
rts1@gsnAboveYRefLineColor = "yellow"
rts1@gsnBelowYRefLineColor = "blue"
rts1@trYMinF = -200.
rts1@trYMaxF = 200.
rts1@tiXAxisString = "Period (days)"
; rts1@tiYAxisString = "Phase (degrees)"
plots(1) = gsn_csm_xy(wks, 1./csdof@frq, csdof@phase, rts1)

; Panel layout
pnlres = True
pnlres@amJust = "TopLeft"

; Draw 1x2 panel
gsn_panel(wks, plots, (/1,2/), pnlres)

print("2019 SM and T PC1 CPSD analysis completed!")

end