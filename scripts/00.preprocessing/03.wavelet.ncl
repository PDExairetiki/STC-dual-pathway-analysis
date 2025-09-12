;load "/mnt/c/ncl/wave_coi/shea_util.ncl"
;load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"    ; Issue with drawing cone of influence

begin

;-------------------------------------------------------------------


f0=addfile("el_paip4icp.nc","r")
x0=f0->paip
tc=dim_avg_n_Wrap(x0,(/2,3/))
tc=dim_standardize_n_Wrap(tc,0,1)
ts=dim_avg_n_Wrap(tc,0)

time=ispan(0,364,1)
N=dimsizes(ts)

;--------------------- Calculate wavelet ------------------------------------

;--------------- Set wavelet parameters, calculate

mother = 2   ;  Morlet wavelet is 0, D.O.G wavelet is 2
dt = 1       ;  Time interval in array, usually 1
param = 6   ;  Mother wavelet parameter, less than 0 for default. For Morlet wavelet, default parameter is 6
s0 = dt      ;  Minimum time scale of wavelet, usually 2*dt. For Morlet wavelet, s0=dt, can resolve 1
dj = 0.25    ;  Common setting
jtot   = 1+floattointeger(((log10(N*dt/s0))/dj)/log10(2.))   ;   Common setting
npad   = N   ;  Common setting
noise  = 1   ;  1 for red noise, 0 for white noise
isigtest  = 0   ;  Use chi-square test; if 1 then test time average of entire spectrum
siglvl = 0.05   ;  Significance test standard
nadof  = 0   ;  Common setting

w = wavelet(ts,mother,dt,param,s0,dj,jtot,npad,noise,isigtest,siglvl,nadof)

;--------------- Wavelet power

power = onedtond(w@power, (/jtot,N/))

power!0 = "period"
power&period = w@period
power!1 = "time"
power&time = time

power@long_name = "power spectrum"
power@units     = "unit^2"

;--------------- Calculate significance ( >= 1 is significant)
 
SIG = power ; Copy metadata 
SIG = power/conform (power,w@signif,0) 
SIG@long_name = "Significance" 
SIG@units = " "

;---------------------  Plot ----------------------------------------

wks = gsn_open_wks("eps", "ts2paip4icp2")
gsn_define_colormap(wks,"sunshine_9lev")

YLValues=(/1,2,4,8,16,32,64,128,256/)
YLLabels=(/"1","2","4","8","16","32","64","128","256"/)

res=True 
res@gsnDraw=False
res@gsnFrame=False 

res@gsnLeftString="(f)D.O.G";Morlet" 

res@trYReverse = True ; Reverse y-axis 

res@tmYLMode = "Explicit" 
res@tmYLValues = YLValues
res@tmYLLabels = YLLabels

res@tmLabelAutoStride = True

;res@trYMaxF = max(YLValues) 
;res@trYMinF = min(YLValues) 

res@cnLevelSelectionMode="ManualLevels"
res@cnMinLevelValF=0.0
res@cnMaxLevelValF=1.0
res@cnLevelSpacingF=0.1

res@cnFillOn = True
res@cnFillMode = "AreaFill" 
res@cnRasterSmoothingOn="True"
res@cnLinesOn = False 
res@cnLineLabelsOn = False 
res@cnInfoLabelOn = False

res@vpHeightF=0.6
res@vpWidthF=0.6

res@lbOrientation = "Vertical"    ; vertical label bar

;res@tiXAxisString = "Day" 
;res@tiXAxisOffsetYF = 0.035 
;res@tiYAxisString = "Period" 

;res@cnRasterSmoothingOn = True
;res@tiMainString   = "Summer Precipitation index morlet wavelet analysis"

res@tmXBMode="Explicit"
res@tmXBValues=(/0,31,59,90,120,151,181,212,243,273,304,334/)
res@tmXBLabels=(/"J","F","M","A","M","J","J","A","S","O","N","D"/)
;res@tmXBValues=(/0,15,30,45,60/)
;res@tmXBLabels=(/"Apr 20th","May 5th","May 20th","Jun 4th","Jun 19th"/)
res@gsnRightString="~F33~p~";"RC";

plot = gsn_csm_contour(wks,power,res)

;--------------- Significance dotting

res2=res

res2@lbLabelBarOn = False
res2@cnFillScaleF          = 2 ; Default is 1, less than 1 increases density (shape filling achieved through ShadeGtContour below)
res2@cnFillDotSizeF = 0.005        ; Thicken dots

iplot = gsn_csm_contour(wks,SIG,res2) 

opt = True 
opt@gsnShadeFillType = "pattern" ; Default setting 
opt@gsnShadeHigh = 17 ; See Appendix Figure A.3 
iplot = gsn_contour_shade(iplot,-999.,1.,opt) ; Fill with pattern 17 starting from first contour greater than or equal to 1.0 overlay(plot,iplot) ; Add significance to original plot
overlay(plot,iplot)

;---------------  Add cone of influence curve at top

plot = ShadeCOI(wks,plot,w,time,True)

;--------------------------------------------------------------------------------------------------------
draw(plot) 
frame(wks)

end