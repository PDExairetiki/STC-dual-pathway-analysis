begin

pathe="/mnt/c/discipulus/doctorial/era/"
pathc="/mnt/c/discipulus/doctorial/cpc/"
path="/mnt/c/discipulus/doctorial/laci/era5/"

f0=addfile(pathe+"el_paip4sa.nc","r")
x0=f0->paip(:,:,:,80:)

;seasonal splitting

x0m=x0(:,59:150,:,:)
x01=dim_avg_n_Wrap(x0m,(/0,1/))

x0j=x0(:,151:242,:,:)
x02=dim_avg_n_Wrap(x0j,(/0,1/))

x0s=x0(:,243:333,:,:)
x03=dim_avg_n_Wrap(x0s,(/0,1/))

x0dec=x0(:39,334:,:,:)
x0jf=x0(1:,:58,:,:)

x0d=new((/40,90,221,181/),float)
x0d(:,:30,:,:)=x0dec
x0d(:,31:,:,:)=x0jf
x04=dim_avg_n_Wrap(x0d,(/0,1/))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

lon=x0&lon
lat=x0&lat
nlon=dimsizes(lon)
nlat=dimsizes(lat)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

wks=gsn_open_wks("png","ann_paip4sa")

stringw=(/"(a)","(b)","(c)","(d)","(e)","(f)","(g)","(h)","(i)","(j)"/)
strings=(/"SM","T","~F33~p~"/)
plots=new(4,graphic)

res=True

res@gsnDraw=False
res@gsnFrame=False

res@sfXCStartV=lon(0)
res@sfXCEndV=lon(nlon-1)
res@sfYCStartV=lat(0)
res@sfYCEndV=lat(nlat-1)

res@mpLimitMode="Corners"
res@mpLeftCornerLonF=lon(0)
res@mpRightCornerLonF=lon(nlon-1)
res@mpLeftCornerLatF=lat(0)
res@mpRightCornerLatF=lat(nlat-1)
res@mpGeophysicalLineThicknessF=2.0

res@mpDataBaseVersion = "Ncarg4_1"
res@mpDataSetName = "Earth..4"
res@mpOutlineBoundarySets = "National"
res@mpOutlineSpecifiers = (/"China:states"/)

res@cnFillOn=True
res@cnLineLabelsOn=False
res@cnFillMode="AreaFill"
res@cnLinesOn=False
res@mpFillOn=True
res@lbLabelBarOn=False

res@tmXTOn=False
res@tmYROn=False
res@tmXBLabelFontHeightF=0.02
res@tmYLLabelFontHeightF=0.02

res@cnLevelSelectionMode="ExplicitLevels"
res@cnLevelSelectionMode = "ManualLevels"; manual set levels so lb consistent
res@cnMinLevelValF       = 0.04             ; min level
res@cnMaxLevelValF       = 0.4              ; max level
res@cnLevelSpacingF      = 0.04             ; contour interval
res@mpLandFillColor      = "white"       ; make land white
res@cnFillColors=(/0,120,110,100,90,80,70,60,50,40,30,20/)
;(/0,120,110,100,90,80,70,60,50,40,30,20/)
;(/20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240/)

res@vpHeightF=0.6
res@vpWidthF=0.4

res@gsnAddCyclic=False

res@gsnLeftString=stringw(0)+"MAM"
res@gsnRightString=strings(0)
plots(0)=gsn_csm_contour_map(wks,x01,res)

res@gsnLeftString=stringw(1)+"JJA"
res@gsnRightString=strings(0)
plots(1)=gsn_csm_contour_map(wks,x02,res)

res@gsnLeftString=stringw(2)+"SON"
res@gsnRightString=strings(0)
plots(2)=gsn_csm_contour_map(wks,x03,res)

res@gsnLeftString=stringw(3)+"DJF"
res@gsnRightString=strings(0)
plots(3)=gsn_csm_contour_map(wks,x04,res)

gres=True
gres@gsLineDashPattern=1
gres@gsLineThicknessF=3

lat1=(/lat(36),lat(116),lat(116),lat(36),lat(36)/)
lon1=(/lon(8),lon(8),lon(77),lon(77),lon(8)/)

lat2=(/lat(80),lat(120),lat(120),lat(80),lat(80)/)
lon2=(/lon(76),lon(76),lon(120),lon(120),lon(76)/)

lat3=(/lat(128),lat(200),lat(200),lat(128),lat(128)/)
lon3=(/lon(8),lon(8),lon(120),lon(120),lon(8)/)

dum0=new(10,graphic)
dum1=new(10,graphic)
dum2=new(10,graphic)

gres@gsLineColor="red"
gressc=gres
gressc@gsLineColor="green"
gresnc=gres
gresnc@gsLineColor="blue"

do i=0,3,1
     dum0(i)=gsn_add_polyline(wks,plots(i),lon1,lat1,gres)
end do

resp=True
resp@gsnDraw=True
resp@gsnFrame=True
resp@gsnPanelLabelBar=True
resp@lbLabelFontHeightF=0.007


gsn_panel(wks,plots,(/2,2/),resp)

end
