begin

path="../../data/climatology/eof/"

; Read SM EOF results
fs=addfile(path+"mam_sml1eof4icp_gms.nc","r")
eof_sm=dim_avg_n_Wrap(fs->eof(:,0,:,:),0)
pcvar_sm=eof_sm@pcvar(0)

; Read T EOF results
ft=addfile(path+"mam_t2m2eof4icp_gms.nc","r")
eof_t=dim_avg_n_Wrap(ft->eof(:,0,:,:),0)
pcvar_t=eof_t@pcvar(0)

; Read STC EOF results
fstc=addfile(path+"mam_paip2eof4icp_gms.nc","r")
eof_stc=dim_avg_n_Wrap(fstc->eof(:,0,:,:),0)
pcvar_stc=eof_stc@pcvar(0)
eof_stc=eof_stc

; Read SM PC1 time series
fse=addfile(path+"mam_sml12ts4icp_gms.nc","r")
ts_sm=dim_avg_n_Wrap(fse->ts(:,0,:),0)

; Read T PC1 time series
fte=addfile(path+"mam_t2m2ts4icp_gms.nc","r")
ts_t=dim_avg_n_Wrap(fte->ts(:,0,:),0)

; Read STC PC1 time series
fstce=addfile(path+"mam_paip2ts4icp_gms.nc","r")
ts_stc=dim_avg_n_Wrap(fstce->ts(:,0,:),0)

lat=eof_sm&lat
lon=eof_sm&lon
nlat=dimsizes(lat)
nlon=dimsizes(lon)

; Create time axis
day=ispan(0,91,1)

wks=gsn_open_wks("png","mam_eof_analysis")

stringw = (/"(a)","(b)","(c)","(d)","(e)","(f)"/)
strings = (/"SM","T","STC - ~F33~p"/)

; Create 6 subplots
plot=new(6,graphic)

; Set map projection parameters
res=True
res@gsnDraw=False
res@gsnFrame=False
res@gsnLeftString=""
res@gsnRightString=""
res@gsnSpreadColors=True
res@gsnAddCyclic=False

res@cnInfoLabelOn=False
res@cnLinesOn=False
res@cnLineLabelsOn=False
res@cnFillOn=True
res@cnFillMode="AreaFill"
res@lbLabelBarOn=True
res@lbOrientation="Horizontal"
res@pmLabelBarOrthogonalPosF=0.2
res@pmLabelBarHeightF=0.1
res@cnLevelSelectionMode="ExplicitLevels"
res@cnLevels=(/-0.01,-0.008,-0.006,-0.004,-0.002,0,0.002,0.004,0.006,0.008,0.01/)
res@cnFillColors=(/32,48,64,80,96,112,144,160,176,192,208,224/)

res@mpMinLatF=lat(0)
res@mpMaxLatF=lat(nlat-1)
res@mpMinLonF=lon(0)
res@mpMaxLonF=lon(nlon-21)
res@mpFillOn=True
res@mpOceanFillColor="gray90"
res@mpLandFillColor="gray90"
res@mpInlandWaterFillColor="gray90"
res@cnMissingValFillColor="gray90"
res@mpShapeMode="FreeAspect"
res@mpDataBaseVersion = "Ncarg4_1"
res@mpDataSetName = "Earth..4"
res@mpOutlineBoundarySets = "National"
res@mpOutlineSpecifiers = (/"China:states"/)

res@tmXTOn=False
res@tmYROn=False
res@tmXBLabelFontHeightF=0.02
res@tmYLLabelFontHeightF=0.02

res@vpHeightF=0.418
res@vpWidthF=0.362

res@pmTickMarkDisplayMode="Always"
res@gsnLeftStringFontHeightF=0.02
res@gsnRightStringFontHeightF=0.02
res@gsnLeftStringOrthogonalPosF=0.02
res@gsnRightStringOrthogonalPosF=0.02

; Plot SM EOF spatial pattern
res@gsnLeftString=stringw(0)+" "+strings(0) 
res@gsnRightString="EOF1 ("+sprintf("%.2f",pcvar_sm)+"%)"
plot(0)=gsn_csm_contour_map_ce(wks,eof_sm,res)

; Plot T EOF spatial pattern
res@gsnLeftString=stringw(1)+" "+strings(1)
res@gsnRightString="EOF1 ("+sprintf("%.2f",pcvar_t)+"%)"
plot(1)=gsn_csm_contour_map_ce(wks,eof_t,res)

; Plot STC EOF spatial pattern
res@gsnLeftString=stringw(2)+" "+strings(2)
res@gsnRightString="EOF1 ("+sprintf("%.2f",pcvar_stc)+"%)"
plot(2)=gsn_csm_contour_map_ce(wks,eof_stc,res)

; Set time series plot parameters
rts=True
rts@gsnDraw=False
rts@gsnFrame=False
rts@gsnScale=True
rts@gsnYRefLine=0.
rts@gsnYRefLineDashPattern=1
rts@gsnYRefLineColor="black"
rts@xyDashPattern=0
rts@xyLineThicknessF=2.

rts@vpHeightF=0.181
rts@vpWidthF=0.362

rts@tiYAxisString=""
rts@trXMinF=min(day)-1
rts@trXMaxF=max(day)+1
rts@trYMinF=-1
rts@trYMaxF=1

rts@tmXTOn=False
rts@tmYROn=False
rts@tmXBLabelFontHeightF=0.02
rts@tmYLLabelFontHeightF=0.02
rts@tmXBMode="Explicit"
rts@tmXBValues=(/0,14,31,45,61,75/)
rts@tmXBLabels=(/"Mar","15th","Apr","15th","May","15th"/)

rts@gsnLeftStringFontHeightF=0.02
rts@gsnRightStringFontHeightF=0.02
rts@gsnLeftStringOrthogonalPosF=0.02
rts@gsnRightStringOrthogonalPosF=0.02

; Plot SM PC1 time series
rts@gsnLeftString=stringw(3)+" "+strings(0)
rts@gsnRightString="PC1"
plot(3)=gsn_csm_xy(wks,day,ts_sm,rts)

; Plot T PC1 time series
rts@gsnLeftString=stringw(4)+" "+strings(1)
rts@gsnRightString="PC1"
plot(4)=gsn_csm_xy(wks,day,ts_t,rts)

; Plot STC PC1 time series
rts@gsnLeftString=stringw(5)+" "+strings(2)
rts@gsnRightString="PC1"
plot(5)=gsn_csm_xy(wks,day,ts_stc,rts)

; Set panel layout
pnlres=True
pnlres@amJust="TopLeft"
pnlres@gsnPanelLabelBar=False

; Draw panel with 2 rows and 3 columns
gsn_panel(wks,plot,(/2,3/),pnlres)

print("Low-latitude Plateau EOF analysis plotting completed!")
print("SM PC1 variance contribution: " + sprintf("%.2f", pcvar_sm) + "%")
print("T PC1 variance contribution: " + sprintf("%.2f", pcvar_t) + "%")
print("STC PC1 variance contribution: " + sprintf("%.2f", pcvar_stc) + "%")

end