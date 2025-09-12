begin

path = "/users/panwei/discipulus/postdoc/0#llp-icp/era/"

f0=addfile(path+"el_paip4icp.nc","r")
x0=f0->paip(:,:,:,:)
x0=dim_standardize_n_Wrap(x0,0,(/0,1/))

xm=x0(:,109:173,:,:)
xmm=dim_avg_n_Wrap(xm,(/0,1/))
tmc=dim_avg_n_Wrap(xm,(/2,3/))
tms=dim_avg_n_Wrap(tmc,0)

xh=x0(40,109:173,:,:)
xhm=dim_avg_n_Wrap(xh,0)
ths=dim_avg_n_Wrap(xh,(/1,2/))
ts0=ths(50:)

f1=addfile(path+"el_t2m4icp.nc","r")
x1=f1->t2m(:,:,:,:)
x1=dim_standardize_n_Wrap(x1,0,(/0,1/))

yh=x1(40,109:173,:,:)
yh=yh*1.6
yhm=dim_avg_n_Wrap(yh,0)

day=ispan(0,64,1)
lat=x0&lat
lon=x0&lon
nlat=dimsizes(lat)
nlon=dimsizes(lon)

wks=gsn_open_wks("png","hw2stc4icp")

stringw=(/"(a)","(b)","(c)","(d)","(e)"/);,"(f)","(g)","(h)","(i)","(j)"
plot=new(3,graphic)

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
res@cnInfoLabelOn=False
res@cnFillOn=True
res@cnFillMode="AreaFill"
res@cnLinesOn=False
res@mpFillOn=True
res@lbLabelBarOn=True
res@cnLevelSelectionMode="ExplicitLevels"
res@cnLevels=(/2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3./);cor4(/0.3,0.4,0.5,0.6,0.7,0.8,0.9,1./);pi4(/-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5/)
res@cnFillColors=(/0,130,140,150,160,170,180,190,200,210,220,230/);cor4(/0,129,145,161,177,193,209,225,241,255/);pi4(/32,48,64,80,96,0,144,160,176,192,208/)
;res@cnLineLabelsOn=False
;res@cnLinesOn=False

res@mpMinLatF=lat(0)
res@mpMaxLatF=lat(nlat-1)
res@mpMinLonF=lon(0)
res@mpMaxLonF=lon(nlon-1)

res@mpFillOn=False
res@mpShapeMode="FreeAspect"
res@mpDataBaseVersion = "Ncarg4_1"
res@mpDataSetName = "Earth..4"
res@mpOutlineBoundarySets = "National"
res@mpOutlineSpecifiers = (/"China:states"/)

res@tmXTOn=False
res@tmYROn=False
res@tmXBLabelFontHeightF=0.02
res@tmYLLabelFontHeightF=0.02

res@vpHeightF=0.6
res@vpWidthF=0.6

res@pmTickMarkDisplayMode="Always"
; res@lbLabelBarOn=False

res@gsnLeftStringFontHeightF=0.02

res@gsnLeftString=stringw(0)+"Heatwave"
res@gsnRightString="T~B~2m~N~";"RC";
plot(0)=gsn_csm_contour_map_ce(wks,yhm,res)

res@gsnLeftString=stringw(1)+"Heatwave"
res@gsnRightString="STC - ~F33~P";"RC";
plot(1)=gsn_csm_contour_map_ce(wks,xhm,res)

gres=True
gres@gsLineDashPattern=1
gres@gsLineThicknessF=3

lat1=(/21,29,29,21,21/)
lon1=(/97,97,106,106,97/)

dum0=new(2,graphic)

gres@gsLineColor="red"

do i=0,1
     dum0(i)=gsn_add_polyline(wks,plot(i),lon1,lat1,gres)
end do

rts=True
rts@gsnDraw=False
rts@gsnFrame=False
rts@gsnScale=True
rts@gsnYRefLine=0.
rts@gsnYRefLineColor="red"
rts@gsnYRefLineDashPattern=2
;rts@gsnXYBarChart=True
;rts@gsnAboveYRefLineColor="red"
;rts@gsnBelowYRefLineColor="blue"

rts@xyLineThicknessF=(/2.,2./)
rts@xyLineColors=(/"red","blue"/)
rts@xyDashPatterns=(/0,0/)

rts@vpHeightF=0.6
rts@vpWidthF=0.6

rts@tiYAxisString=""

rts@trXMinF=min(day)-1
rts@trXMaxF=max(day)+1
rts@trYMinF=-3
rts@trYMaxF=3

rts@tmXTOn=False
rts@tmYROn=False
rts@tmXBLabelFontHeightF=0.02
rts@tmYLLabelFontHeightF=0.02
rts@tmXBMode="Explicit"
rts@tmXBValues=(/11,25,42,56/);(/0,31,59,90,120,151,181,212,243,273,303,334/)
rts@tmXBLabels=(/"May 1st","May 15th","Jun 1st","Jun 15th"/);(/"Mar","15th","Apr","15th","May","15th","Jun","15th"/);(/"J","F","M","A","M","J","J","A","S","O","N","D"/)

rts@gsnLeftStringFontHeightF=0.02
;rts@tiXAxisString="Day"

ts00=(/ths,tms/)
rts@gsnLeftString=stringw(3)+"Regional Mean"
rts@gsnRightString="STC - ~F33~p";"RC";
plot(2)=gsn_csm_xy(wks,day,ts00,rts)

res_text=True
res_text@txFontHeightF=0.015
res_text@txJust="CenterLeft"

res_lines=True 
res_lines@gsLineDashPattern=0.
res_lines@gsLineThicknessF=5.
res_lines@gsLineColor="red"
res_li=res_lines
res_sc=res_lines
res_sc@gsLineColor="blue"

xx=(/5,15/);
yy0li=(/-1,-1/);
yy0sc=(/-1.4,-1.4/);
dumli01=gsn_add_polyline(wks,plot(2),xx,yy0li,res_li)
dumsc01=gsn_add_polyline(wks,plot(2),xx,yy0sc,res_sc)
dumli02=gsn_add_text(wks,plot(2),"Heatwave",20,yy0li,res_text)
dumsc02=gsn_add_text(wks,plot(2),"Climatology",20,yy0sc,res_text)


pnlres=True
pnlres@amJust="TopLeft"

gsn_panel(wks,plot,(/1,3/),pnlres)

end
